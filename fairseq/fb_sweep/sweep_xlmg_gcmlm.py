#!/usr/bin/env python
"""
Example usage:

    PYTHONPATH=. ./fb_sweep/xlmg/sweep_xlmg_gcmlm.py \
            --num-trials 1 --num-gpus 8 --num-nodes 1 \
            --model-size 125M \
            --prefix xlmg.125m \
            --variant MLM \
            --partition learnaccel

This sweep script takes some additional optional arguments. See add_extra_options_func
for more details.
"""

import os
from metaseq_internal.fb_sweep.sweep import (
    hyperparam,
    get_env_from_args,
    main as fb_sweep_main,
)
from metaseq_internal.constants import (
    BASE_MODEL_UNSHARDED_EN_DATA_LOCATIONS as UNSHARDED_EN_DATA,
    ComputeEnvs,
)


def add_extra_options_func(parser):
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="use synthetic data and only train for 50 steps (for benchmarking)",
    )
    parser.add_argument(
        "--model-size",
        required=True,
        help="model configuration, see get_grid for available options",
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "MLM",
            "CLM",
            "CLM_prefix",
            "CM3",
            "CM3_prefix",
            "CM3_masked_only",
            "CLM_MLM",
        ],
    )
    parser.add_argument("--seq-len", type=int, default=1024, help="tokens_per_sample")
    parser.add_argument(
        "--restore-file", help="load an existing checkpoint for continuing training"
    )
    parser.add_argument(
        "--debug-train-on-small-subset",
        action="store_true",
        help="only load a single shard of data from one datasource (OpenWebText), "
        "which reduces startup time and is useful for debugging",
    )
    parser.add_argument(
        "--optimizer",
        "--opt",
        default="adam",
        choices=["adam", "adam8bit", "cpu_adam"],
        help="which optimizer to use",
    )
    parser.add_argument("--scale-attn", action="store_true", default=False)
    parser.add_argument("--scale-fc", action="store_true", default=False)
    parser.add_argument("--scale-heads", "--sh", action="store_true")
    parser.add_argument("--lr", default=None, type=float, help="overrides default lr")
    parser.add_argument("--no-fp16-adam", action="store_true", default=False)
    parser.add_argument(
        "--bs", default=None, type=int, help="overrides default local batch size"
    )
    parser.add_argument(
        "--no-ckpt",
        default=False,
        action="store_true",
        help="dont checkpoint activations",
    )
    parser.add_argument(
        "--stable", default=False, action="store_true", help="use StableEmbeddingLayer"
    )
    parser.add_argument("--alibi", default=False, action="store_true")
    parser.add_argument("--use-fused-softmax", default=False, action="store_true")
    parser.add_argument("--scale-resids", default=False, action="store_true")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--save-interval", default=1000, type=int)
    parser.add_argument("--dropout", default=None)
    parser.add_argument("--end-learning-rate", default=None, type=float)
    parser.add_argument("--zero-lr-warmup-steps", default=None, type=int)
    parser.add_argument(
        "--zero2",
        action="store_true",
        help="use ZeRO-2 instead of ZeRO-3, which speeds up training by ~5% at the "
        "cost of more memory usage; ideal for dense equiv. models <10B params",
    )
    parser.add_argument(
        "--clip-norm-type",
        default="l2",
        choices=["l2", "inf"],
        help="norm for grad clipping",
    )


def get_base_model_config(layers, model_dim, heads):
    return [
        hyperparam("--arch", "transformer_lm_gpt", save_dir_key=lambda val: val),
        hyperparam("--activation-fn", "gelu"),
        hyperparam("--share-decoder-input-output-embed"),
        hyperparam("--decoder-layers", layers, save_dir_key=lambda val: f"nlay{val}"),
        hyperparam(
            "--decoder-embed-dim", model_dim, save_dir_key=lambda val: f"emb{val}"
        ),
        hyperparam("--decoder-ffn-embed-dim", 4 * model_dim),
        hyperparam("--decoder-attention-heads", heads),
    ]


def add_moe_config_(args, model_config, expert_count):
    model_config.extend(
        [
            # general config
            hyperparam(
                "--max-sentences-valid", 1
            ),  # not strictly necessary, but safer to avoid OOM
            hyperparam("--num-workers-valid", 0),  # this can avoid hangs in some cases
            # options exposed in model
            hyperparam(
                "--moe-expert-count",
                expert_count,
                save_dir_key=lambda val: f"nexprt{val}",
            ),
            hyperparam("--moe-freq", 2),  # MOE on every other layer
            hyperparam("--moe-gating-use-fp32"),
            hyperparam("--moe-second-expert-policy", "all"),
            # hyperparam("--moe-normalize-gate-prob-before-dropping", save_dir_key=lambda val: "norm_b"),
            hyperparam(
                "--moe-eval-capacity-token-fraction", -1.0
            ),  # use same capacity during valid and train
            # options exposed in criterion
            hyperparam("--criterion", "moe_cross_entropy"),
            hyperparam(
                "--moe-gate-loss-wt", [0.01], save_dir_key=lambda val: f"moe_w{val}"
            ),
            hyperparam("--moe-gate-loss-combine-method", "sum"),
            hyperparam(
                "--moe-normalize-expert-grad",
                "sqrt_world_size",
                save_dir_key=lambda val: val,
            ),
        ]
    )
    if not args.benchmark:
        model_config.extend(
            [
                hyperparam("--pad-to-fixed-length"),
            ]
        )


def add_adam8bit_config_(optimizer_config):
    optimizer_config.extend(
        [
            hyperparam("--use-sharded-state"),
            hyperparam("--stable-emb"),
            hyperparam("--no-scale-embedding"),
            hyperparam("--block-wise"),
        ]
    )


def add_cpu_adam_config_(optimizer_config):
    optimizer_config.extend(
        [
            hyperparam("--optimizer", "cpu_adam"),
            hyperparam("--cpu-offload", save_dir_key=lambda _: "cpuoff"),
            hyperparam("--offload-activations", save_dir_key=lambda _: "offloadact"),
        ]
    )


def get_grid(args):
    num_gpus = args.num_gpus * args.num_nodes
    training_tokens = int(100e9)  # GPT-3 used 300e9

    # Set this to 0 on AWS to avoid segfaults
    num_dataloading_workers = 2 if not os.path.exists("/fsx") else 0

    if args.debug_train_on_small_subset:
        train_subset = "train13"
        assert args.prefix.startswith(
            "test"
        ), "please ensure that --prefix starts with 'test' when using --debug-train-on-small-subset"
    else:
        train_subset = "train"

    # TODO the original dense training runs in H1 2021 used a single validation
    # set coming from CC-News. If you need to have comparable valid_ppl to those
    # runs, then set this to False. Otherwise True is preferred, since it will
    # aggregate all of the valid sets for CC-News, Books, Wikipedia, etc.
    combine_valid_sets = True
    cluster_env = get_env_from_args(args)

    if args.data is None:
        args.data = UNSHARDED_EN_DATA[cluster_env]
    for shard in args.data.split(":"):
        assert os.path.exists(shard), f"Could not find data path: {shard}"

    if cluster_env == ComputeEnvs.AWS or cluster_env == ComputeEnvs.AZURE or args.local:
        args.snapshot_code = False  # containers don't support snapshot_code
    else:
        args.snapshot_code = True

    # Model configuration based on size
    M = 1024 * 1024
    if args.model_size == "tiny":
        model_config = get_base_model_config(layers=2, model_dim=128, heads=4)
        batch_size_tokens = int(8 * 1024)
        max_batch_size_per_gpu = 16
        learning_rate = 6e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01
    elif args.model_size == "125M":
        # assert num_gpus >= 8
        model_config = get_base_model_config(layers=12, model_dim=768, heads=12)
        batch_size_tokens = int(0.5 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 6e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01
    elif args.model_size == "355M":
        assert num_gpus >= 8
        model_config = get_base_model_config(layers=24, model_dim=1024, heads=16)
        batch_size_tokens = int(0.5 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 3e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01
    elif args.model_size == "1.3B":
        assert num_gpus >= 32
        model_config = get_base_model_config(layers=24, model_dim=2048, heads=32)
        batch_size_tokens = int(1.0 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 2e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01
    elif args.model_size == "2.7B":
        assert num_gpus >= 64
        model_config = get_base_model_config(layers=32, model_dim=2560, heads=32)
        batch_size_tokens = int(1.0 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 1.6e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01
    elif args.model_size == "6.7B":
        assert num_gpus >= 64
        model_config = get_base_model_config(layers=32, model_dim=4096, heads=32)
        batch_size_tokens = int(2.0 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 1.2e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01
    elif args.model_size == "13B":
        assert num_gpus >= 64
        model_config = get_base_model_config(layers=40, model_dim=5120, heads=40)
        batch_size_tokens = int(2.0 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 1.0e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01
    else:
        raise ValueError(f"Unknown --model-size argument: {args.model_size}")

    if args.dropout is not None:
        dropout = dropout

    # Batch size logic
    batch_size_seqs = batch_size_tokens // args.seq_len
    if args.bs is not None:
        max_batch_size_per_gpu = args.bs
    batch_size_per_gpu = min(max_batch_size_per_gpu, batch_size_seqs // num_gpus)
    update_freq = batch_size_seqs // (batch_size_per_gpu * num_gpus)
    assert (
        batch_size_tokens == update_freq * batch_size_per_gpu * num_gpus * args.seq_len
    )

    max_update = training_tokens // batch_size_tokens
    warmup_updates = warmup_tokens // batch_size_tokens

    log_interval = 10 if not args.local else 1

    if args.benchmark:
        # Overrides for speed benchmarking
        args.data = None
        task_config = [
            hyperparam("--task", "dummy_lm", save_dir_key=lambda val: val),
            hyperparam(
                "--tokens-per-sample",
                args.seq_len,
                save_dir_key=lambda val: f"tps{val}",
            ),
            hyperparam("--dict-size", 51200 - 4),
            hyperparam("--disable-validation"),
        ]
        max_update = 50
        warmup_updates = 50
        log_interval = 5
    else:
        clm_prob = 0.0
        mask_prob = 0.15
        fully_causal_prob = 0.0
        fully_bidir_prob = 0.0
        predict_masked_only = False
        if args.variant == "MLM":
            fully_bidir_prob = 1.0
        elif args.variant == "CLM":
            clm_prob = 1.0
        elif args.variant == "CLM_prefix":
            clm_prob = 0.1
            mask_prob = 0.0
        elif args.variant == "CLM_MLM":
            clm_prob = 0.5
            fully_bidir_prob = 1.0
        elif args.variant == "CM3":
            clm_prob = 0.1
            fully_causal_prob = 1.0
        elif args.variant == "CM3_prefix":
            clm_prob = 0.1
        elif args.variant == "CM3_masked_only":
            predict_masked_only = True
            fully_causal_prob = 1.0
        else:
            raise NotImplementedError(f"Unknown variant {args.variant}")
        task_config = [
            hyperparam("--task", "gcmlm"),
            hyperparam("--clm-prob", clm_prob, save_dir_key=lambda val: f"clm{val}"),
            hyperparam("--mask-prob", mask_prob, save_dir_key=lambda val: f"mask{val}"),
            hyperparam(
                "--fully-causal-prob",
                fully_causal_prob,
                save_dir_key=lambda val: f"causal{val}",
            ),
            hyperparam(
                "--fully-bidir-prob",
                fully_bidir_prob,
                save_dir_key=lambda val: f"bidir{val}",
            ),
            hyperparam(
                "--sample-break-mode", "complete", save_dir_key=lambda val: f"bm_{val}"
            ),
            hyperparam(
                "--tokens-per-sample",
                args.seq_len,
                save_dir_key=lambda val: f"tps{val}",
            ),
            hyperparam("--skip-invalid-size-inputs-valid-test"),
        ]
        if predict_masked_only:
            task_config.append(
                hyperparam(
                    "--predict-masked-only", save_dir_key=lambda val: "masked_only"
                )
            )

    # Optimizer config
    optimizer = args.optimizer
    optimizer_config = [
        hyperparam("--optimizer", optimizer, save_dir_key=lambda val: val)
    ]
    if not args.no_fp16_adam and optimizer != "adam8bit":
        optimizer_config.append(
            hyperparam("--fp16-adam-stats", save_dir_key=lambda val: "fp16adam")
        )
    if optimizer == "adam":
        pass  # defaults set elsewhere
    elif optimizer == "adam8bit":
        add_adam8bit_config_(model_config)
    elif optimizer == "cpu_adam":
        optimizer_config.extend(
            [
                hyperparam("--fp16-adam-stats", save_dir_key=lambda val: "fp16adam"),
            ]
        )
        add_cpu_adam_config_(model_config)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    grid = []

    def H(*args, **kwargs):
        """Add a hyperparameter"""
        grid.append(hyperparam(*args, **kwargs))

    H("--use-sharded-state")

    if args.stable:
        H("--stable-emb", True, binary_flag=True, save_dir_key=lambda x: "stable_emb")
        H("--no-scale-embedding")

    if args.restore_file:
        grid += [
            hyperparam("--restore-file", args.restore_file),
        ]
    if combine_valid_sets:
        grid += [hyperparam("--combine-val")]
    else:
        grid += [hyperparam("--ignore-unused-valid-subsets")]
    grid += [
        hyperparam("--train-subset", train_subset),
        hyperparam("--num-workers", num_dataloading_workers),
        hyperparam("--validate-interval-updates", 1000),
        hyperparam("--save-interval-updates", args.save_interval),
        # hyperparam("--no-epoch-checkpoints"),  # only save checkpoints based on num steps
        hyperparam("--no-best-checkpoints"),  # don't save checkpoint_best.pt
        hyperparam(
            "--keep-interval-updates", 1
        ),  # only keep the most recent checkpoint
        # hyperparam("--no-save-optimizer-state-on-training-finished"),
        # hyperparam("--save-async"),
        hyperparam("--ddp-backend", "fully_sharded", save_dir_key=lambda val: "fsdp"),
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--fp16-init-scale", 4),
        hyperparam("--threshold-loss-scale", 0.25),
    ]

    if not args.no_ckpt:
        H("--checkpoint-activations")

    if args.zero2:
        grid += [
            hyperparam("--no-reshard-after-forward", save_dir_key=lambda val: "zero2")
        ]
    grid += model_config
    grid += task_config
    grid += optimizer_config

    lr_to_use = learning_rate if args.lr is None else args.lr
    grid += [
        # GPT-3 uses "(0.9, 0.95)"
        hyperparam(
            "--adam-betas",
            "(0.9, 0.98)",
            save_dir_key=lambda val: "b2_{}".format(eval(val)[1]),
        ),
        # Sometimes lowering --adam-eps to 1e-6 can stabilize training
        hyperparam(
            "--adam-eps", args.eps, save_dir_key=lambda val: f"eps{val}"
        ),  # GPT-3 used --clip-norm=1.0
        hyperparam("--clip-norm", 1.0, save_dir_key=lambda val: f"cl{val}"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", lr_to_use, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", max_update),
        hyperparam(
            "--warmup-updates", warmup_updates, save_dir_key=lambda val: f"wu{val}"
        ),
        hyperparam("--dropout", dropout, save_dir_key=lambda val: f"dr{val}"),
        hyperparam(
            "--attention-dropout", dropout, save_dir_key=lambda val: f"atdr{val}"
        ),
        hyperparam("--weight-decay", weight_decay, save_dir_key=lambda val: f"wd{val}"),
        # hyperparam("--max-tokens", batch_size_per_gpu*args.seq_len, save_dir_key=lambda val: f"maxtoks{val}"),  # HACK Use this for complete_doc
        hyperparam(
            "--batch-size", batch_size_per_gpu, save_dir_key=lambda val: f"ms{val}"
        ),
        hyperparam("--required-batch-size-multiple", 1),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--seed", 1, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", log_interval),
        # hyperparam("--wandb-project", "xlmg-architecture-gcmlm"),
    ]

    if args.end_learning_rate is not None:
        H(
            "--end-learning-rate",
            args.end_learning_rate,
            save_dir_key=lambda x: f"end_lr_{x}",
        )
    if args.zero_lr_warmup_steps is not None:
        H(
            "--zero-lr-warmup-steps",
            args.zero_lr_warmup_steps,
            save_dir_key=lambda x: f"0lr_wu{x}",
        )
    H(
        "--scale-attn",
        args.scale_attn,
        binary_flag=True,
        save_dir_key=lambda x: "ln_attn" if x else "",
    )
    H(
        "--scale-fc",
        args.scale_fc,
        binary_flag=True,
        save_dir_key=lambda x: "ln_fc" if x else "",
    )
    H(
        "--scale-heads",
        args.scale_heads,
        binary_flag=True,
        save_dir_key=lambda x: "scale_heads" if x else "",
    )
    H(
        "--use-fused-softmax",
        args.use_fused_softmax,
        binary_flag=True,
        save_dir_key=lambda x: "fused" if x else "",
    )
    H(
        "--scale-resids",
        args.scale_resids,
        binary_flag=True,
        save_dir_key=lambda x: "scale_resids" if x else "",
    )
    H(
        "--alibi",
        args.alibi,
        binary_flag=True,
        save_dir_key=lambda x: "alibi" if x else "",
    )
    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    # Check hyperparam value in config.keys(), to avoid mis-specifying in get_grid.
    if "--clip-norm-type" in config.keys():
        norm_type = config["--clip-norm-type"].current_value
        assert norm_type in [
            "l2",
            "inf",
        ], f"Invalid --clip-norm-type of {norm_type}! Only 'l2' and 'inf' supported!"


if __name__ == "__main__":
    fb_sweep_main(
        get_grid, postprocess_hyperparams, add_extra_options_func=add_extra_options_func
    )
