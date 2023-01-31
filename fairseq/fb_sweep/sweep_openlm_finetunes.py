#!/usr/bin/env python
"""
This sweep script takes some additional optional arguments. See add_extra_options_func
for more details.

To use this script, run

```
python -m fb_sweep.sweep_openlm_finetunes --azure --model-size 175b -g 8 -n 8 --checkpoints-dir <checkpoint path> --fine-tune-type dialogue -p <name of sweep>
```

If things break part of the way through:

1. If `checkpoint_last` is working properly (ex, dataloader loads at the same epoch for everything):
    * Run the same command as above with the `--resume-failed` flag.
2. If `checkpoint_last` is broken:
    * Delete `checkpoint_last` in the relevant directory
    * Add `--restore-file <path to last good checkpoint>` and `--resume-failed` to the command above.

"""
import os

from metaseq_internal.constants import (
    TOTAL_TRAIN_TOKENS,
    MODEL_SIZES,
    DATA_LOCATIONS,
    ComputeEnvs,
)
from metaseq_internal.fb_sweep.sweep import (
    hyperparam,
    get_env_from_args,
    main as fb_sweep_main,
)

# assert all sizes make sense, as the gpt-3 paper contains typos
for name, size in MODEL_SIZES.items():
    assert size.n_heads * size.d_head == size.emb_size, name

# have to do this at the module level, unfortunately; unable to use args.<env>
for _cluster, _folder in DATA_LOCATIONS.items():
    if os.path.exists(_folder):
        if _cluster != ComputeEnvs.RSC:
            from metaseq_internal.fb_sweep.dependency_checks import *  # noqa
        break


FINE_TUNE_CONFIGS = (
    {  # assumes prefix matching "DATA_LOCATIONS" above unless full path specified
        "loading_test": {
            "path": "loading_test",
            "total_tokens": 100,
        },
        "dialogue": {
            "path": "ft_dialogue",
            "total_tokens": 2.3e8,
        },
        "pretraining": {
            "path": "gptz/corpus_dedup_10_10_1_0.05_exp29",
            "total_tokens": TOTAL_TRAIN_TOKENS,
        },
        "dialogue_and_flan": {
            "path": "/data/home/mpchen/real/ft_dialogue_and_flan",
            "total_tokens": 4.7e8,
        },
    }
)

PRETRAIN_MODEL_LOCATIONS = {
    ComputeEnvs.AWS: {
        "175b": "/fsx-mudslide/sshleifer/checkpoints/175B_model_ws512/reshard.pt",
    },
    ComputeEnvs.AZURE: {"175b": "/data/xlmg/models/175B_ws512/reshard.pt"},
}


def add_extra_options_func(parser):
    # NOTE we shouldn't add new options here... track changes via git instead
    parser.add_argument(
        "--finetune-from-model",
        help="load an existing checkpoint for initial fine-tuning",
    )
    parser.add_argument(
        "--restore-file", help="load an existing checkpoint for continuing training"
    )
    parser.add_argument("--model-size", choices=MODEL_SIZES.keys(), required=True)
    parser.add_argument(
        "--no-save-dir", action="store_true", help="avoid saving with hparams"
    )
    parser.add_argument(
        "--verbose-save-dir", default=False, help="store verbose hparam name"
    )
    parser.add_argument(
        "--fine-tune-type",
        choices=FINE_TUNE_CONFIGS.keys(),
        required=True,
        help="which fine-tune are we doing?",
    )
    parser.add_argument(
        "--limit-valid-steps",
        default=True,
        help="limit the # of validation steps to not take forever. There's an additional 'MAX_VALID_SECONDS' that can also be mucked with here that we have not yet instrumented as an argument, but is set to 3600 at the moment.",
    )


def get_grid(args):
    cluster_env = get_env_from_args(args)
    DATA_ROOT = DATA_LOCATIONS[cluster_env].replace("/gptz", "")

    # Infer data path if not given
    if args.data is None:
        # Where is our fine-tuning data
        path = FINE_TUNE_CONFIGS[args.fine_tune_type]["path"][0]
        if path[0] == "\\":
            args.data = path  # for hard coded paths when we're testing datasets locally
        else:
            args.data = os.path.join(
                DATA_ROOT, FINE_TUNE_CONFIGS[args.fine_tune_type]["path"]
            )
        # Finagle valid files a bit so that they'll show up per valid set
        valid_files = os.listdir(os.path.join(args.data, "valid"))
        valid_files = ",".join([f"valid/{x}/" for x in valid_files])
        print(valid_files)
    else:
        print(args.data)

    if args.finetune_from_model is None and args.restore_file is None:
        args.finetune_from_model = PRETRAIN_MODEL_LOCATIONS[cluster_env][
            args.model_size
        ]

    SEQ_LEN = 2048
    size = MODEL_SIZES[args.model_size]
    # Equations from the pretraining to use as reference. Note that we fix `ddp_bsz` to make determining which batchsize OOMs on AWS easier. `ddp_bsz` can be thought of as the batchsize per GPU. On AWS, 8 with FP16 is the highest power of 2 that does not OOM; 4 is the highest power of 2 without FP 16.
    # See https://docs.google.com/document/d/1xeUJ3eeF0mHfmkq6Mb4Onnx_Y-EvPJ9PH2bKd5ughxw/edit?pli=1#heading=h.z6pt13778lcb for context
    # updates = 300B tokens / 2048 seq_len / 1024 batchsize
    # ddp_bsz = (size.batch_size // total_gpus) // SEQ_LEN
    total_gpus = (args.num_gpus * args.num_nodes) // size.model_parallel
    ddp_bsz = 8
    effective_batch_size = ddp_bsz * SEQ_LEN * total_gpus
    dataset_train_tokens = FINE_TUNE_CONFIGS[args.fine_tune_type]["total_tokens"]
    updates = int((3 * dataset_train_tokens) // effective_batch_size)
    warmup_updates = int((dataset_train_tokens * 0.1) // effective_batch_size)

    learning_rates = [3e-6, 6e-6, 1e-5, 3e-5, 6e-5]

    # as a convenience for finding things later on
    args.snapshot_root = args.checkpoints_dir + "_fairseq-snapshot"
    args.snapshot_code = True

    def save_name(is_verbose=True):
        if args.no_save_dir:
            return False
        if is_verbose:
            return args.verbose_save_dir
        return True

    grid = [
        hyperparam("--train-subset", "train"),
        hyperparam("--valid-subset", valid_files),
        hyperparam("--ignore-unused-valid-subsets"),
        hyperparam("--num-workers", 8),
        hyperparam("--num-workers-valid", 8),
        hyperparam(
            "--validate-interval-updates", 200
        ),  # takes about a minute per update for fine tuning
        hyperparam("--save-interval-updates", 800),
        hyperparam(
            "--no-epoch-checkpoints"
        ),  # Save at least a *little* bit of a memory when we're working with gigantic models; validate_interval_updates + save_interval_updates should be frequent enough to get close enough.
        hyperparam(
            "--memory-efficient-fp16",
            save_dir_key=lambda val: "me_fp16" if save_name() else "",
        ),
        hyperparam("--fp16-init-scale", 4),
        # we set this for the main run but it's probably not needed here. Including so we've got a log between the hyperparams that are different between here and there.
        # hyperparam("--threshold-loss-scale", 0.25, save_dir_key=lambda val: f"minscale{val}"),
        hyperparam(
            "--ddp-backend",
            "fully_sharded",
            save_dir_key=lambda val: "fsdp" if save_name() else "",
        ),
        hyperparam("--use-sharded-state"),
        hyperparam("--checkpoint-activations"),
        hyperparam("--model-parallel-size", size.model_parallel),
        hyperparam("--criterion", "vocab_parallel_cross_entropy"),
        hyperparam("--distribute-checkpointed-activations"),
        # Flags to match exact same initialization of Megatron code for exp 12.00
        hyperparam("--full-megatron-init"),
        hyperparam("--megatron-init-sigma", 0.006),
        hyperparam(
            "--activation-fn", "relu", save_dir_key=lambda x: x if save_name() else ""
        ),
        hyperparam(
            "--arch",
            "transformer_lm_megatron",
            save_dir_key=lambda val: val if save_name() else "",
        ),
        hyperparam("--share-decoder-input-output-embed"),
        hyperparam(
            "--decoder-layers",
            size.n_layers,
            save_dir_key=lambda val: f"nlay{val}" if save_name() else "",
        ),
        hyperparam(
            "--decoder-embed-dim",
            size.emb_size,
            save_dir_key=lambda val: f"emb{val}" if save_name() else "",
        ),
        hyperparam("--decoder-ffn-embed-dim", size.ffn_size),
        hyperparam("--decoder-attention-heads", size.n_heads),
        # Switch to learned position embeddings for exp 12.00, without scaling
        hyperparam(
            "--decoder-learned-pos",
            save_dir_key=lambda _: "lrnpos" if save_name() else "",
        ),
        hyperparam(
            "--no-scale-embedding",
            save_dir_key=lambda _: "0emb_scale" if save_name() else "",
        ),
        hyperparam("--task", "streaming_language_modeling"),
        hyperparam(
            "--sample-break-mode",
            "none",
            save_dir_key=lambda val: f"bm_{val}" if save_name() else "",
        ),
        hyperparam(
            "--tokens-per-sample",
            SEQ_LEN,
            save_dir_key=lambda val: f"tps{val}" if save_name() else "",
        ),
        hyperparam(
            "--vocab-filename",
            os.path.join(DATA_ROOT, "gptz/tokenizers/gpt2-vocab.json"),
            save_dir_key=lambda _: "gpt2" if save_name() else "",
        ),
        hyperparam(
            "--merges-filename",
            os.path.join(DATA_ROOT, "gptz/tokenizers/gpt2-merges.txt"),
        ),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        # GPT-3 uses "(0.9, 0.95)"
        hyperparam(
            "--adam-betas",
            f"(0.9, 0.95)",
            save_dir_key=lambda val: "b2_{}".format(eval(val)[1])
            if save_name()
            else "",
        ),
        # Sometimes lowering --adam-eps to 1e-6 can stabilize training
        hyperparam(
            "--adam-eps",
            1e-8,
            save_dir_key=lambda val: f"eps{val}" if save_name() else "",
        ),
        # GPT-3 used --clip-norm=1.0
        hyperparam(
            "--clip-norm",
            0.2,
            save_dir_key=lambda val: f"cl{val}" if save_name() else "",
        ),
        hyperparam("--clip-norm-type", "l2"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam(
            "--lr",
            learning_rates,
            save_dir_key=lambda val: f"lr{val:.3g}"
            if save_name(is_verbose=False)
            else "",
        ),
        hyperparam(
            "--end-learning-rate",
            3e-7,
            save_dir_key=lambda val: f"endlr{val:.3g}"
            if save_name(is_verbose=False)
            else "",
        ),
        hyperparam(
            "--warmup-updates",
            warmup_updates,
            save_dir_key=lambda val: f"wu{val}" if save_name(is_verbose=False) else "",
        ),
        hyperparam("--total-num-update", updates),
        hyperparam(
            "--dropout", 0.1, save_dir_key=lambda val: f"dr{val}" if save_name() else ""
        ),
        hyperparam(
            "--attention-dropout",
            0.1,
            save_dir_key=lambda val: f"atdr{val}" if save_name() else "",
        ),
        hyperparam(
            "--no-emb-dropout", save_dir_key=lambda _: "0emb_dr" if save_name() else ""
        ),
        hyperparam(
            "--weight-decay",
            0.1,
            save_dir_key=lambda val: f"wd{val}" if save_name() else "",
        ),
        hyperparam(
            "--batch-size",
            ddp_bsz,
            save_dir_key=lambda val: f"ms{val}" if save_name(is_verbose=False) else "",
        ),
        # Emperically determined to be the most that can be handelled...
        hyperparam(
            "--batch-size-valid",
            2,
            save_dir_key=lambda val: f"ms{val}" if save_name(is_verbose=False) else "",
        ),
        hyperparam(
            "--update-freq",
            1,
            save_dir_key=lambda val: f"uf{val}" if save_name() else "",
        ),
        hyperparam(
            "--max-update",
            updates,
            save_dir_key=lambda val: f"mu{val}" if save_name() else "",
        ),
        hyperparam(
            "--seed", 1, save_dir_key=lambda val: f"seed{val}" if save_name() else ""
        ),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 1),
        hyperparam("--required-batch-size-multiple", 1),
        hyperparam("--gradient-predivide-factor", 32),
        hyperparam("--tensor-parallel-init-model-on-gpu"),
        hyperparam("--threshold-loss-scale", 0.25),
        hyperparam("--fp16-adam-stats", save_dir_key=lambda val: "fp16adam"),
    ]
    if args.limit_valid_steps:
        MAX_VALID_SECONDS = 3600
        valid_datasets = len(valid_files.split(","))
        seconds_per_step = 75.0 / 2  # determined emperically
        valid_steps_per_dataset = int(
            (MAX_VALID_SECONDS / seconds_per_step) / valid_datasets
        )
        print("validating each dataset for", valid_steps_per_dataset, "steps")
        grid += [hyperparam("--max-valid-steps", valid_steps_per_dataset)]

    if args.restore_file:
        grid += [hyperparam("--restore-file", args.restore_file)]
    elif args.finetune_from_model:
        grid += [hyperparam("--finetune-from-model", args.finetune_from_model)]

    return grid


def postprocess_hyperparams(args, config):
    pass


if __name__ == "__main__":
    fb_sweep_main(
        get_grid, postprocess_hyperparams, add_extra_options_func=add_extra_options_func
    )
