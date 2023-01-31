#!/usr/bin/env python
"""
This file reproduces the 350m model training log parameter file

`/checkpoint/ktirumala/optlogs/gptz-baselines/350m/logs/train.log`

when using the 350M model size, when you run the following command:


"""

from metaseq_internal.fb_sweep.dependency_checks import *  # noqa
from metaseq_internal.fb_sweep.sweep import hyperparam, main as fb_sweep_main

from metaseq_internal.constants import (
    MODEL_SIZES,
)

def add_extra_options_func(parser):
    # NOTE we shouldn't add new options here... track changes via git instead
    parser.add_argument(
        "--restore-file", help="load an existing checkpoint for continuing training"
    )

    print(MODEL_SIZES)

    parser.add_argument(
        "--model-size",
        choices=MODEL_SIZES.keys(),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )


def get_grid(args):
    # Infer data path if not given
    if args.data is None:
        # Rotated data used for 11.7+ experiments
        # args.data = "/data/xlmg/gptz/corpus_dedup_10_10_1_0.05_run11.7/"
        # Returning back to original dataset for 12.00 onward.
        # args.data = "/data/xlmg/gptz/corpus_dedup_10_10_1_0.05_exp29"
        print("Please give a data directory")
        return

    # Fix number of training updates (this should match 300B tokens)
    TARGET_TOKENS = 300*10**9

    TOKENS_PER_SAMPLE = 2048
    NGPUS = args.num_gpus * args.num_nodes

    # Seems to be a discrepancy between theoretical values and observed WPB, maybe due to padding? Anyway, account for that.
    # TODO increase this
    TOKEN_OVERHEAD = 1.0

    TOTAL_UPDATES = int(TARGET_TOKENS * TOKEN_OVERHEAD // args.batch_size // TOKENS_PER_SAMPLE // NGPUS)

    # Use the standard OPT model sizes
    model_size = MODEL_SIZES[args.model_size]

    # Use all valid sets available
    valid_files = ",".join([f"valid/{x}" for x in os.listdir(os.path.join(args.data, "valid"))])
    print(valid_files)

    args.snapshot_code = True
    grid = [
        hyperparam("--train-subset", "train"),
        #hyperparam("--valid-subset", ",".join(f"valid/{ss}" for ss in VALID_SUBSETS)),
        hyperparam("--valid-subset", valid_files),
        hyperparam("--ignore-unused-valid-subsets"),
        hyperparam("--num-workers", 8),
        hyperparam("--num-workers-valid", 1),
        # kill training if it doesn't make progress in X seconds
        # hyperparam("--heartbeat-timeout", 3600),
        hyperparam("--validate-interval-updates", 50000),
        hyperparam("--save-interval-updates", 50000),
        hyperparam("--keep-interval-updates", 1),
        hyperparam(
            "--no-epoch-checkpoints"
        ),  # only save checkpoints based on num steps
        hyperparam("--no-best-checkpoints"),  # don't save checkpoint_best.pt
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--fp16-init-scale", 4),
        hyperparam(
            "--threshold-loss-scale", 0.25, save_dir_key=lambda val: f"minscale{val}"
        ),
        hyperparam("--ddp-backend", "pytorch_ddp", save_dir_key=lambda val: "fsdp"),
        # hyperparam("--use-sharded-state"),
        # hyperparam(
        #     "--gradient-predivide-factor", 32.0, save_dir_key=lambda x: f"gpf{x}"
        # ),
        # hyperparam("--checkpoint-activations"),
        # hyperparam("--model-parallel-size", 1),
        # hyperparam("--criterion", "cross_entropy"),
        # hyperparam("--distribute-checkpointed-activations"),
        # NOTE: we didn't start training with this, we are adding it now for restarts with restore file cause the
        # initialization doesn't matter and it will be much faster.
        # TODO: Add this back when we do restores. Commenting out for cold start on exp 12.00.
        # Adding back in for restart for exp 12.03.
        # hyperparam("--tensor-parallel-init-model-on-gpu"),
        # Flags to match exact same initialization of Megatron code for exp 12.00
        # hyperparam("--full-megatron-init"),
        # hyperparam("--megatron-init-sigma", 0.006),
        hyperparam("--activation-fn", "relu", save_dir_key=lambda x: x),
        hyperparam("--arch", "transformer_lm", save_dir_key=lambda val: val),
        hyperparam("--share-decoder-input-output-embed"),

        # Kushal's mini-run
        # hyperparam("--decoder-layers", 6, save_dir_key=lambda val: f"nlay{val}"),
        # hyperparam("--decoder-embed-dim", 128, save_dir_key=lambda val: f"emb{val}"),
        # hyperparam("--decoder-ffn-embed-dim", 4 * 128),
        # hyperparam("--decoder-attention-heads", 4),

        # Standard OPT sizes
        hyperparam("--decoder-layers", model_size.n_layers, save_dir_key=lambda val: f"nlay{val}"),
        hyperparam("--decoder-embed-dim", model_size.emb_size, save_dir_key=lambda val: f"emb{val}"),
        hyperparam("--decoder-ffn-embed-dim", model_size.ffn_size),
        hyperparam("--decoder-attention-heads", model_size.n_heads),

        # hyperparam("--decoder-learned-sinusoidal", save_dir_key=lambda _: "lrnsin"),
        # Switch to learned position embeddings for exp 12.00, without scaling
        hyperparam("--decoder-learned-pos", save_dir_key=lambda _: "lrnpos"),
        hyperparam("--no-scale-embedding", save_dir_key=lambda _: "0emb_scale"),
        # Remove normformer for exp 12.00 (was there in 11.xx experiments)
        # hyperparam("--scale-fc", save_dir_key=lambda _: "nffc"),
        # hyperparam("--scale-attn", save_dir_key=lambda _: "nfatt"),
        # hyperparam("--scale-heads", save_dir_key=lambda _: "nfhd"),
        hyperparam("--task", "streaming_language_modeling"),
        hyperparam("--sample-break-mode", "none", save_dir_key=lambda val: f"bm_{val}"),
        hyperparam("--tokens-per-sample", TOKENS_PER_SAMPLE, save_dir_key=lambda val: f"tps{val}"),
        hyperparam(
            "--vocab-filename",
            "/datasets01/gptz_corpus_dedup_10_10_1_0.05_exp29/120321/tokenizers/gpt2-vocab.json",
            save_dir_key=lambda _: "gpt2",
        ),
        hyperparam("--merges-filename", "/datasets01/gptz_corpus_dedup_10_10_1_0.05_exp29/120321/tokenizers/gpt2-merges.txt"),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        # GPT-3 uses "(0.9, 0.95)"
        hyperparam(
            "--adam-betas",
            f"(0.9, 0.98)",
            save_dir_key=lambda val: "b2_{}".format(eval(val)[1]),
        ),
        # Sometimes lowering --adam-eps to 1e-6 can stabilize training
        hyperparam("--adam-eps", 1e-8, save_dir_key=lambda val: f"eps{val}"),
        # GPT-3 used --clip-norm=1.0
        hyperparam("--clip-norm", 1.0, save_dir_key=lambda val: f"cl{val}"),
        # Skip gradient updates when hitting clip norm threshold.
        # This flag will reject most updates at start of training so hard threshold are bad at start of training.
        # Since we are adding in between the run 11.6, it should be hopefully okay.
        # Removed in 11.9.
        # hyperparam("--skip-gradient-update-on-clip-norm"),
        hyperparam("--clip-norm-type", "l2"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", model_size.lr, save_dir_key=lambda val: f"lr{val:.3g}"),
        # hyperparam(
        #     "--end-learning-rate", 6.0e-6, save_dir_key=lambda val: f"endlr{val}"
        # ),
        hyperparam("--warmup-updates", 2000, save_dir_key=lambda val: f"wu{val}"),

        hyperparam("--total-num-update", TOTAL_UPDATES),
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--no-emb-dropout", save_dir_key=lambda _: "0emb_dr"),
        hyperparam("--weight-decay", 0.01, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--batch-size", args.batch_size, save_dir_key=lambda val: f"ms{val}"),
        hyperparam("--update-freq", 1, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", TOTAL_UPDATES, save_dir_key=lambda val: f"mu{val}"),

        # args.seed is by default 1
        hyperparam("--seed", 1, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 1),
        hyperparam("--required-batch-size-multiple", 1),
    ]
    if args.restore_file:
        grid += [hyperparam("--restore-file", args.restore_file)]
    # if args.reset_dataloader:
    #     grid += [hyperparam("--reset-dataloader")]

    return grid


def postprocess_hyperparams(args, config):
    pass


if __name__ == "__main__":
    fb_sweep_main(
        get_grid, postprocess_hyperparams, add_extra_options_func=add_extra_options_func
    )
