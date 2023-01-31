#!/usr/bin/env python
"""
This is a reference document for the run12.00 setup.
"""

from metaseq_internal.fb_sweep.dependency_checks import *  # noqa
from metaseq_internal.fb_sweep.sweep import hyperparam, main as fb_sweep_main

VALID_SUBSETS = [
    # standard corpora
    "BookCorpusFair",
    "CommonCrawl",
    "DM_Mathematics",
    "Gutenberg_PG-19",
    "HackerNews",
    "OpenSubtitles",
    "OpenWebText2",
    "USPTO",
    "Wikipedia_en",
    "redditflattened",
    "stories",
    # dialogue datasets. NOT included in "everything"
    "dialogue_chitchat",  # BST + Convai2 + Empathetic Dialogues
    "dialogue_knowledge",  # wiz of wiki, wiz of int
    "dialogue_tod",  # metalwoz, taskmaster2, google_sgd, multiwoz
    "dialogue_light",  # light_dialog_wild
    # "dialogue_safety",  # recovering from safety failures
]


def add_extra_options_func(parser):
    # NOTE we shouldn't add new options here... track changes via git instead
    parser.add_argument(
        "--restore-file", help="load an existing checkpoint for continuing training"
    )
    parser.add_argument(
        "--reset-dataloader",
        action="store_true",
        help="reset the dataloader to epoch 1",
    )


def get_grid(args):
    # Infer data path if not given
    if args.data is None:
        # Rotated data used for 11.7+ experiments
        # args.data = "/data/xlmg/gptz/corpus_dedup_10_10_1_0.05_run11.7/"
        # Returning back to original dataset for 12.00 onward.
        args.data = "/data/xlmg/gptz/corpus_dedup_10_10_1_0.05_exp29"

    args.snapshot_code = True
    grid = [
        hyperparam("--train-subset", "train"),
        hyperparam("--valid-subset", ",".join(f"valid/{ss}" for ss in VALID_SUBSETS)),
        hyperparam("--ignore-unused-valid-subsets"),
        hyperparam("--num-workers", 8),
        hyperparam("--num-workers-valid", 1),
        # kill training if it doesn't make progress in X seconds
        # hyperparam("--heartbeat-timeout", 3600),
        hyperparam("--validate-interval-updates", 1000),
        hyperparam("--save-interval-updates", 250),
        hyperparam(
            "--no-epoch-checkpoints"
        ),  # only save checkpoints based on num steps
        hyperparam("--no-best-checkpoints"),  # don't save checkpoint_best.pt
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--fp16-init-scale", 4),
        hyperparam(
            "--threshold-loss-scale", 0.25, save_dir_key=lambda val: f"minscale{val}"
        ),
        hyperparam("--ddp-backend", "fully_sharded", save_dir_key=lambda val: "fsdp"),
        hyperparam("--use-sharded-state"),
        hyperparam(
            "--gradient-predivide-factor", 32.0, save_dir_key=lambda x: f"gpf{x}"
        ),
        hyperparam("--checkpoint-activations"),
        hyperparam("--model-parallel-size", 8),
        hyperparam("--criterion", "vocab_parallel_cross_entropy"),
        hyperparam("--distribute-checkpointed-activations"),
        # NOTE: we didn't start training with this, we are adding it now for restarts with restore file cause the
        # initialization doesn't matter and it will be much faster.
        # TODO: Add this back when we do restores. Commenting out for cold start on exp 12.00.
        # Adding back in for restart for exp 12.03.
        hyperparam("--tensor-parallel-init-model-on-gpu"),
        # Flags to match exact same initialization of Megatron code for exp 12.00
        hyperparam("--full-megatron-init"),
        hyperparam("--megatron-init-sigma", 0.006),
        hyperparam("--activation-fn", "relu", save_dir_key=lambda x: x),
        hyperparam("--arch", "transformer_lm_megatron", save_dir_key=lambda val: val),
        hyperparam("--share-decoder-input-output-embed"),
        hyperparam("--decoder-layers", 96, save_dir_key=lambda val: f"nlay{val}"),
        hyperparam("--decoder-embed-dim", 12288, save_dir_key=lambda val: f"emb{val}"),
        hyperparam("--decoder-ffn-embed-dim", 4 * 12288),
        hyperparam("--decoder-attention-heads", 96),
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
        hyperparam("--tokens-per-sample", 2048, save_dir_key=lambda val: f"tps{val}"),
        hyperparam(
            "--vocab-filename",
            "/data/xlmg/gptz/tokenizers/gpt2-vocab.json",
            save_dir_key=lambda _: "gpt2",
        ),
        hyperparam("--merges-filename", "/data/xlmg/gptz/tokenizers/gpt2-merges.txt"),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        # GPT-3 uses "(0.9, 0.95)"
        hyperparam(
            "--adam-betas",
            f"(0.9, 0.95)",
            save_dir_key=lambda val: "b2_{}".format(eval(val)[1]),
        ),
        # Sometimes lowering --adam-eps to 1e-6 can stabilize training
        hyperparam("--adam-eps", 1e-8, save_dir_key=lambda val: f"eps{val}"),
        # GPT-3 used --clip-norm=1.0
        hyperparam("--clip-norm", 0.3, save_dir_key=lambda val: f"cl{val}"),
        # Skip gradient updates when hitting clip norm threshold.
        # This flag will reject most updates at start of training so hard threshold are bad at start of training.
        # Since we are adding in between the run 11.6, it should be hopefully okay.
        # Removed in 11.9.
        # hyperparam("--skip-gradient-update-on-clip-norm"),
        hyperparam("--clip-norm-type", "l2"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", 3e-5, save_dir_key=lambda val: f"lr{val:.3g}"),
        hyperparam(
            "--end-learning-rate", 6.0e-6, save_dir_key=lambda val: f"endlr{val}"
        ),
        hyperparam("--warmup-updates", 2000, save_dir_key=lambda val: f"wu{val}"),
        # updates = 300B tokens / 2048 seq_len / 1024 bsz
        hyperparam("--total-num-update", 143052),
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--no-emb-dropout", save_dir_key=lambda _: "0emb_dr"),
        hyperparam("--weight-decay", 0.1, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--batch-size", 8, save_dir_key=lambda val: f"ms{val}"),
        hyperparam("--update-freq", 1, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", 143052, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--seed", 1, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 1),
        hyperparam("--required-batch-size-multiple", 1),
    ]
    if args.restore_file:
        grid += [hyperparam("--restore-file", args.restore_file)]
    if args.reset_dataloader:
        grid += [hyperparam("--reset-dataloader")]

    return grid


def postprocess_hyperparams(args, config):
    pass


if __name__ == "__main__":
    fb_sweep_main(
        get_grid, postprocess_hyperparams, add_extra_options_func=add_extra_options_func
    )
