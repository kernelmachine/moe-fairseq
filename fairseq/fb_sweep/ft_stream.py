#!/usr/bin/env python
"""
Launch streaming language model fine-tuning.
"""
import os

from fairseq.fb_sweep.sweep import (
    hyperparam,
    main as fb_sweep_main,
)
from fairseq.constants import MODEL_SIZES


DEFAULT_RANDOM_SEED = 1234


def get_grid(args):
    grid = []

    def H(*args, **kwargs):
        grid.append(hyperparam(*args, **kwargs))

    size = MODEL_SIZES[args.model_size]

    if args.restore_file:
        H("--restore-file", args.restore_file, save_dir_key=lambda _: args.model_size)
    if args.fast_forward:
        H("--fast-forward", args.fast_forward)

    grid += [
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--no-best-checkpoints"),
        hyperparam("--save-interval", args.save_interval_updates),
        hyperparam("--save-interval-updates", args.save_interval_updates),
        hyperparam("--keep-interval-updates", args.keep_interval_updates),
        hyperparam("--validate-interval-updates", args.interval),
    ]
    if args.validate_at_beginning:
        grid += [hyperparam("--validate-at-beginning")]
    if args.no_save:
        H("--no-save")
    else:
        H("--best-checkpoint-metric", "loss")

    H("--task", "streaming_finetune_language_modeling")
    H("--vocab-filename", args.vocab_filename)
    H("--merges-filename", args.merges_filename)
    H("--sample-break-mode", args.sbm)

    if args.valid_subset == "valid":
        H(
            "--combine-valid-subsets"
        )  # this by default assumes the split name as valid
    else:
        H(
            "--valid-subset", args.valid_subset
        )  # valid sets are separated by comma and given as a string
    
    
    H(
        "--train-subset", args.train_subset
    )  # train sets are separated by comma and given as a string

    assert (
        args.tps == 2048
    ), "Fix required to allow loading learned positional embeddings with different ws"
    H("--criterion", "moe_cross_entropy")

    H("--arch", "transformer_lm_gpt3_xl")
    H("--activation-fn", "relu")
    H("--decoder-learned-pos")
    H("--share-decoder-input-output-embed")

    # MOE Hyperparams
    H("--moe-expert-count", args.moe_num_experts, save_dir_key=lambda val: f"nexperts_{val}")
    H("--moe-freq", 2)
    H("--moe-gating-use-fp32")
    if args.moe_initialize_from_opt:
        H("--moe-initialize-from-opt", args.moe_initialize_from_opt)
    if args.moe_path_to_expert_state_dict:
        H("--moe-path-to-expert-state-dict", args.moe_path_to_expert_state_dict)
    H("--moe-second-expert-policy", "all")
    H("--moe-normalize-expert-grad", "sqrt_world_size")
    H("--moe-eval-capacity-token-fraction", -1.0)
    H("--moe-gate-loss-wt", 0.01)
    H("--moe-gate-loss-combine-method", "sum")

    if not args.embdr:
        H("--no-emb-dropout", save_dir_key=lambda _: "0edr")
    if args.min_loss_scale > 0:
        H("--min-loss-scale", args.min_loss_scale)
    # Add document attention seperator to efficiently finetune under streaming setting.
    if args.self_attn_doc_sep:
        H("--self-attn-doc-sep", 2, save_dir_key=lambda val: f"docsep_{val}")
    H("--checkpoint-activations", binary_flag=True) 
    H("--decoder-learned-pos")
    H("--no-scale-embedding")
    H("--tokens-per-sample", args.tps) 
    H("--ddp-backend", "fully_sharded")

    if args.max_valid_steps > 0:
        H("--max-valid-steps", args.max_valid_steps)

    grid.extend(
        [
            hyperparam("--decoder-layers", size.n_layers),
            hyperparam("--decoder-embed-dim", size.emb_size),
            hyperparam("--decoder-ffn-embed-dim", size.ffn_size),
            hyperparam("--decoder-attention-heads", size.n_heads),
            hyperparam("--share-decoder-input-output-embed"),
        ]
    )

    grid += [
        hyperparam("--max-update", args.max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--total-num-update", args.max_update),
        hyperparam("--warmup-updates", args.warmup_update, save_dir_key=lambda val: f"wu{val}"),
        hyperparam("--required-batch-size-multiple", 1),
        hyperparam("--batch-size", args.bs, save_dir_key=lambda val: f"bsz{val}"),
        # Use a fixed batch size for valid. Since we limit the max valid steps,
        # the number of valid examples should be consistent across different hyperparam
        hyperparam("--batch-size-valid", 2),
        hyperparam("--update-freq", args.uf, save_dir_key=lambda val: f"uf{val}"),
    ]

    # regularization
    dropout = args.dropout
    grid += [
        hyperparam("--dropout", dropout), 
        # --attention-dropout will be set to mirror --dropout in postprocess_args
        hyperparam(
            "--attention-dropout", dropout),
    ]
    if args.wd > 0:
        H("--weight-decay", args.wd, save_dir_key=lambda val: f"wd{val}")
    is_175B = args.model_size == "175b"
    H("--adam-betas", "(0.9, 0.95)")
    H("--adam-eps", 1e-6)
    H("--clip-norm", args.clip_norm, save_dir_key=lambda val: f"clip{val}" if args.clip_norm < 1.0 else "")
    if not args.no_fp16_adam:
        H("--optimizer", "adam", save_dir_key=lambda val: "fp16adam")
    else:
        H("--optimizer", "adam", save_dir_key=lambda val: "fp32adam")

    # random seed
    grid += [
        hyperparam("--seed", args.random_seed, save_dir_key=lambda val: f"rs{val}")
    ]

    H("--memory-efficient-fp16")

    # lr scheduler
    grid += [
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", args.lr, save_dir_key=lambda val: f"lr{val}"),
    ]
    H("--end-learning-rate", args.end_learning_rate, save_dir_key=lambda val: f"endlr{val:.3g}" if args.end_learning_rate !=0 else "")

    if args.bf16:
        H("--fp16")  # this need to be set for bf16
        H("--bf16", save_dir_key=lambda _: "bf16")
    else:
        H("--fp16")

    H("--fp16-init-scale", 128)

    # data loading settings
    H("--num-workers", args.nw)
    H("--num-workers-valid", args.nw)

    # logging settings
    H("--log-format", "json")
    H("--log-interval", 10)
    if args.no_zero3:
        H("--no-reshard-after-forward")
    H("--patience", args.patience, save_dir_key=lambda val: f"pat_{val}")
    if args.wandb_project is not None:
        H("--wandb-project", args.wandb_project)

    return grid


def postprocess_hyperparams(args, config):
    pass


def add_args(parser):
    parser.add_argument("--model-size", choices=MODEL_SIZES.keys(), required=True)
    parser.add_argument(
        "--finetune-from-model",
        help="load an existing checkpoint for initial fine-tuning",
    )
    parser.add_argument(
        "--restore-file", help="load an existing checkpoint for continuing training"
    )
    parser.add_argument(
        "--fast-forward", help="fastforward to future epoch"
    )
    parser.add_argument("--data-type", type=str, default=None)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--cheat", action="store_true")
    parser.add_argument(
        "--random-seed", type=int, nargs="+", default=[DEFAULT_RANDOM_SEED]
    )
    parser.add_argument("--right-trunc", action="store_true")
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-5])
    parser.add_argument("--no-fp16-adam", action="store_true")
    parser.add_argument("--valid-subset", type=str, default="valid")
    parser.add_argument("--train-subset", type=str, default="train")

    parser.add_argument("--max-update", "--mu", type=int, default=None)
    parser.add_argument("--tps", "--seq-len", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--end-learning-rate", type=float, default=0.0)
    parser.add_argument("--uf", type=int, default=1)
    parser.add_argument("--bs", type=int, nargs="+", default=[8])
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--warmup-update", type=int, default=60)
    parser.add_argument("--interval", type=int, default=10000000)
    parser.add_argument("--save-interval-updates", type=int, default=10000000)
    parser.add_argument("--keep-interval-updates", type=int, default=1)

    parser.add_argument("--validate-at-beginning", action="store_true")
    parser.add_argument("--no-zero3", action="store_true")
    parser.add_argument("--patience", type=int, default=10000)
    parser.add_argument("--min-loss-scale", type=float, default=-1)
    parser.add_argument("--sbm", type=str, default="none")
    parser.add_argument("--nw", type=int, default=0)

    parser.add_argument("--moe-num-experts", type=int)
    parser.add_argument("--moe-initialize-from-opt", type=str, default="")
    parser.add_argument("--label-loss", action="store_true")
    parser.add_argument("--embdr", action="store_true")
    parser.add_argument("--eps", type=int, nargs="+", default=[-1])
    parser.add_argument("--self-attn-doc-sep", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-valid-steps", type=int, default=-1)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--pretrain-data-sampling-prob", type=float, default=0.0)

if __name__ == "__main__":
    fb_sweep_main(get_grid, postprocess_hyperparams, add_extra_options_func=add_args)