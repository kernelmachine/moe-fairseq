#!/usr/bin/env python
"""
Launch streaming language model fine-tuning on all environments.

Example command (grid search):
    OUTPUT_CHECKPOINT_DIR=/checkpoint/$USER/instruct_models
    NUM_NODES=8
    MU=1000
    INTERVAL=200

    python -m metaseq_internal.fb_sweep.ft_stream -n $NUM_NODES -g 8 -p finetune.allbenchmarks \
        --checkpoints-dir $OUTPUT_CHECKPOINT_DIR --model-size 13b --data-type all_bench_v4 \
        --lr 1e-5 3e-5 5e-5 --max-update $MU --eps 64 256 --interval $INTERVAL \
        --save-interval-updates $INTERVAL --no-fp16-adam --bs 2 8 --sbm none --label-loss \
        --partition learnfair,learnaccel --resume-failed --fair \
        --max-valid-steps 100 ## temporary for speed testing; TODO: remove later 
"""
import os

from fairseq.fb_sweep.sweep import (
    hyperparam,
    get_env_from_args,
    main as fb_sweep_main,
)
from fairseq.constants import MODEL_SIZES, DATA_LOCATIONS, ComputeEnvs, VALID_SUBSETS


DEFAULT_RANDOM_SEED = 1234

# have to do this at the module level, unfortunately; unable to use args.<env>
for _cluster, _folder in DATA_LOCATIONS.items():
    if os.path.exists(_folder):
        if _cluster != ComputeEnvs.RSC:
            from fairseq.fb_sweep.dependency_checks import *  # noqa
        break

DATA_LOCATIONS[ComputeEnvs.FAIR] = "/gscratch/zlab/sg01/data/"
# DATA_LOCATIONS[ComputeEnvs.FAIR] = "/datasets01/"
DATA_LOCATIONS[ComputeEnvs.AWS] = "/fsx-mudslide/rpasunuru/data"
DATA_LOCATIONS[ComputeEnvs.AZURE] = "/shared/home/sviyer/data"

PRETRAIN_MODEL_LOCATIONS = {
    ComputeEnvs.FAIR: {
        "125m": "/gscratch/zlab/sg01/opt/125m/checkpoint_last.pt",
        "350m": "/checkpoint/victorialin/opt_models/350M_gptz/checkpoint_last.pt",
        "1.3b": "/gscratch/zlab/sg01/opt/1.3b/checkpoint_last.pt",
        "resharded_1.3b": "/checkpoint/suching/opt_models/resharded_1.3b/checkpoint_last.pt",
        "6.7b": "/checkpoint/victorialin/opt_models/6.7B_gptz/checkpoint_last.pt",
        #"1.3b": "/checkpoint/suching/opt_ft/dense/finetune.opt.s2orc_data.1.3b.0edr.mu10000.wu0.bsz8.fp16adam.rs1234.lr2e-05.pat_10.ngpu4/checkpoint_1_10000.pt",
        #"6.7b": "/checkpoint/suching/opt_ft/dense/1_cluster/finetune.opt.opt_data.6.7b.0edr.mu10000.wu0.bsz8.fp16adam.rs1234.lr1.2e-05.pat_100.ngpu4/checkpoint_3_10000.pt",
        
        "13b": "/checkpoint/rpasunuru/opt_models/13B_gptz/checkpoint_last.pt",
        "30b": "/checkpoint/victorialin/opt_models/30B_gptz/checkpoint_last.pt",
        "175b": "/checkpoint/rpasunuru/opt_models/175B_ws512_no_os/reshard.pt",
    },
    ComputeEnvs.AWS: {
        "1.3b": "/fsx-mudslide/xlmg/models/1.3B/checkpoint_last.pt",
        "13b": "/fsx-mudslide/xlmg/models/13B_gptz/checkpoint_last.pt",
        "30b": "/fsx-mudslide/xlmg/models/30B_gptz/checkpoint_last.pt",
        "175b": "/fsx-mudslide/xlmg/models/175B_ws512_no_os/reshard.pt",
    },
    ComputeEnvs.AZURE: {
        "1.3b": "/data/gpt-z/models/gptz/1.3B/raw/checkpoint_last.pt",
        "13b": "/data/gpt-z/models/gptz/13B/raw/checkpoint_last.pt",
        "30b": "/data/gpt-z/models/gptz/30B/raw/checkpoint_last.pt",
        "175b": "/shared/home/sviyer/175B_ws512_no_os/175B_ws512_no_os/reshard.pt",
    },
}

VALID_SUBSETS_2000 = [
    "valid_cause_effect_classification",
    "valid_cause_effect_classification_fs5",
    "valid_data_to_text",
    "valid_data_to_text_fs5",
    "valid_mmlu_fs5",
]

META_ICL_VALID_SUBSETS = [
    "valid_cause_effect_classification",
    "valid_cause_effect_classification_fs5",
    "valid_data_to_text",
    "valid_data_to_text_fs5",
    "valid_dialogue_generation",
    "valid_dialogue_generation_fs5",
    "valid_question_answering",
    "valid_question_answering_fs5",
    "valid_sentence_composition",
    "valid_sentence_composition_fs5",
    "valid_stereotype_detection",
    "valid_stereotype_detection_fs5",
    "valid_summarization",
    "valid_toxic_language_detection",
    "valid_toxic_language_detection_fs5",
    "valid_mmlu_fs5",
]

FINE_TUNE_DATA_CONFIGS = {
    "all_bench_v2": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v2",
    },
    "all_bench_v3": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v3",
    },
    "all_bench_v4": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v4",
    },
    "all_bench_v4_sorted_deleted": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v4_sorted_deleted",
    },
    "all_bench_v4_with_pt": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v4_with_pt",
    },
    "test_dataset": {
        "path": "prompt_data/test_dataset",
    },
    "all_bench_v5": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v5",
    },
    "all_bench_v5_sorted": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v5_sorted",
    },
    "all_bench_v5_sorted_with_pt": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v5_sorted_with_pt",
    },
    "all_bench_v6_sorted": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v6_sorted",
        "valid_subsets": META_ICL_VALID_SUBSETS,
    },
    "all_bench_v6_sorted_2000": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v6_sorted_2000",
        "valid_subsets": VALID_SUBSETS_2000,
    },
    "all_bench_v6_sorted_2000_zipf_a2_max32": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v6_sorted_2000_metaicl_zipf_a2_max32",
        "valid_subsets": VALID_SUBSETS_2000,
    },
    "all_bench_v6_sorted_zipf_a2_max32": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v6_sorted_metaicl_zipf_a2_max32",
        "valid_subsets": META_ICL_VALID_SUBSETS,
    },
    "all_bench_v6_sorted_zipf_a1.00001_max32": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v6_sorted_metaicl_zipf_a1.00001_max32",
        "valid_subsets": META_ICL_VALID_SUBSETS,
    },
    "all_bench_v6_sorted_k-shot_k5": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v6_sorted_metaicl_k-shot_k5",
        "valid_subsets": META_ICL_VALID_SUBSETS,
    },
    "all_bench_v6_sorted_k-shot_k32": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v6_sorted_metaicl_k-shot_k32",
        "valid_subsets": META_ICL_VALID_SUBSETS,
    },
    "all_bench_v6_sorted_uniform_max32": {
        "path": "prompt_data/allbenchmarks_io_streaming_after_dedup_v6_sorted_metaicl_uniform_max32",
        "valid_subsets": META_ICL_VALID_SUBSETS,
    },
    "flan_data": {
        "path": "instruct-opt/prompt_data/flan_io_streaming/",
        "valid_subsets": ["valid"],
        "train_subsets": ["train"]

    },
    "opt_data": {
        # make sure to change DATA_LOCATIONS above
        "path": "opt/120321/",
        "valid_subsets": ["valid/C4_small"],
        "train_subsets": ["train/C4"]
    },
    "s2orc_data": {
        # make sure to change DATA_LOCATIONS above
        "path": "s2orc/",
        "valid_subsets": ["valid"],
        "train_subsets": ['train']
    },
    "pile_data_alt": {
        "path": "pile_data",
        "valid_subsets": ["valid/CommonCrawl"],
        "train_subsets": ['train']
    },
    "demix_data": {
        "path": "demix_data",
        "valid_subsets": ["valid"],
        "train_subsets": ['train']
    },
    "stories": {
        "path": "pile_data",
        "valid_subsets": ["valid/stories"],
        "train_subsets": ['stories']
    },
    "DM_Mathematics": {
        "path": "pile_data",
        "valid_subsets": ["valid/DM_Mathematics"],
        "train_subsets": ['DM_Mathematics']
    },
    "Gutenberg_PG-19": {
        "path": "pile_data",
        "valid_subsets": ["valid/Gutenberg_PG-19"],
        "train_subsets": ['Gutenberg_PG-19']
    },
    "OpenWebText2": {
        "path": "pile_data",
        "valid_subsets": ["valid/OpenWebText2"],
        "train_subsets": ['OpenWebText2']
    },
    "Wikipedia_en": {
        "path": "pile_data",
        "valid_subsets": ["valid/Wikipedia_en"],
        "train_subsets": ['Wikipedia_en']
    },
    "redditflattened": {
        "path": "pile_data",
        "valid_subsets": ["valid/redditflattened"],
        "train_subsets": ['redditflattened']
    },
    "HackerNews": {
        "path": "pile_data",
        "valid_subsets": ["valid/HackerNews"],
        "train_subsets": ['HackerNews']
    },
    "CommonCrawl": {
        "path": "pile_data",
        "valid_subsets": ["valid/CommonCrawl"],
        "train_subsets": ['CommonCrawl']
    },
    "BookCorpusFair": {
        "path": "opt/120321/",
        "valid_subsets": ["valid/BookCorpusFair"],
        "train_subsets": ['train/BookCorpusFair']
    },
    "Enron_Emails": {
        "path": "opt/120321/",
        "valid_subsets": ["valid/Enron_Emails"],
        "train_subsets": ['train/Enron_Emails']
    },
    "OpenSubtitles": {
        "path": "opt/120321/",
        "valid_subsets": ["valid/OpenSubtitles"],
        "train_subsets": ['train/OpenSubtitles']
    },
    "USPTO": {
        "path": "opt/120321/",
        "valid_subsets": ["valid/USPTO"],
        "train_subsets": ['train/USPTO']
    },
    "ccnewsv2": {
        "path": "opt/120321/",
        "valid_subsets": ["valid/CommonCrawl"],
        "train_subsets": ['train/ccnewsv2']
    },
    "1b": {
        "path": "demix_data/1b/",
        "valid_subsets": ["valid"],
        "train_subsets": ['train']
    },
    "cs": {
        "path": "demix_data/cs/",
        "valid_subsets": ["valid"],
        "train_subsets": ['train']
    },
    "med": {
        "path": "demix_data/med/",
        "valid_subsets": ["valid"],
        "train_subsets": ['train']
    },
    "legal": {
        "path": "demix_data/legal/",
        "valid_subsets": ["valid"],
        "train_subsets": ['train']
    },
    "openwebtext": {
        "path": "demix_data/anonymized_openwebtext/",
        "valid_subsets": ["valid"],
        "train_subsets": ['train']
    },
    "realnews": {
        "path": "demix_data/anonymized_realnews/",
        "valid_subsets": ["valid"],
        "train_subsets": ['train']
    },
    "reviews": {
        "path": "demix_data/anonymized_reviews/",
        "valid_subsets": ["valid"],
        "train_subsets": ['train']
    },
    "reddit": {
        "path": "demix_data/reddit/",
        "valid_subsets": ["valid"],
        "train_subsets": ['train']
    },
    "imdb": {
        "path": "imdb/",
        "valid_subsets": ["valid"],
        "train_subsets": ['train']
    },
    "c4": {
        "path": "c4/",
        "valid_subsets": ["valid"],
        "train_subsets": ['train']
    },

}

    


# python -m  fairseq_cli.train \
#     $DATA_DIR  \
#     --save-dir $SAVE_DIR \
#     --ddp-backend fully_sharded \
#     --memory-efficient-fp16 \
#     --checkpoint-activations   \
#     --task streaming_finetune_language_modeling \
#     --tokens-per-sample 2048   \
#     --arch transformer_lm_gpt3_xl \
#     --share-decoder-input-output-embed   \
#     --moe-expert-count $NUM_EXPERTS \
#     --moe-freq 2   \
#     --moe-gating-use-fp32 \
#     --moe-second-expert-policy all   \
#     --moe-normalize-expert-grad sqrt_world_size   \
#     --moe-eval-capacity-token-fraction -1.0  \
#     --max-sentences-valid 1 \
#     --num-workers-valid 0   \
#     --criterion moe_cross_entropy \
#     --moe-gate-loss-wt 0.01 \
#     --moe-gate-loss-combine-method sum   \
#     --optimizer adam \
#     --save-interval-updates 1000 \
#     --lr-scheduler polynomial_decay \
#     --fp16-adam-stats \
#     --adam-betas '(0.9, 0.95)' \
#     --clip-norm 1.0   \
#     --lr $LR \
#     --weight-decay 0.1 \
#     --batch-size 8 \
#     --update-freq 1   \
#     --total-num-update 10000 \
#     --max-update 10000 \
#     --disable-validation   \
#     --log-format json \
#     --log-interval 10 \
#     --train-subset train \
#     --merges-filename /gscratch/zlab/sg01/opt/vocab/gpt2-merges.txt \
#     --vocab-filename /gscratch/zlab/sg01/opt/vocab/gpt2-vocab.json \
#     --reset-dataloader \
#     --reset-meters

def get_grid(args):
    grid = []
    cluster_env = get_env_from_args(args)
    DATA_ROOT = DATA_LOCATIONS[cluster_env]

    def H(*args, **kwargs):
        grid.append(hyperparam(*args, **kwargs))

    if args.data is None:
        if args.data_type is None:
            raise Exception(
                f"Either args.data or args.data_type arguments must be set. Available data_type(s): FINE_TUNE_DATA_CONFIGS.keys()"
            )
        assert args.data_type in FINE_TUNE_DATA_CONFIGS
        semi_path = FINE_TUNE_DATA_CONFIGS[args.data_type]["path"]
        args.data = os.path.join(DATA_ROOT, semi_path)
        # check if given valid subsets exist otherwise select all valid subsets
        if "valid_subsets" in FINE_TUNE_DATA_CONFIGS[args.data_type]:
            avail_valid_subsets = FINE_TUNE_DATA_CONFIGS[args.data_type]["valid_subsets"]
        else:
            avail_valid_subsets = [
                f.name for f in os.scandir(args.data) if f.is_dir() and "valid" in f.name
            ]
        if args.valid_subset not in avail_valid_subsets:
            args.valid_subset = ",".join(avail_valid_subsets)
        if "train_subsets" in FINE_TUNE_DATA_CONFIGS[args.data_type]:
            avail_train_subsets = FINE_TUNE_DATA_CONFIGS[args.data_type]["train_subsets"]
        else:
            avail_train_subsets = [
                f.name for f in os.scandir(args.data) if f.is_dir() and "train" in f.name
            ]
        if args.train_subset not in avail_train_subsets:
            args.train_subset = ",".join(avail_train_subsets)

    size = MODEL_SIZES[args.model_size]
    # if args.finetune_from_model is None and args.restore_file is None:
    #     args.finetune_from_model = PRETRAIN_MODEL_LOCATIONS[cluster_env][
    #         args.model_size
    #     ]

    if args.restore_file:
        H("--restore-file", args.restore_file, save_dir_key=lambda _: args.model_size)
    # elif args.finetune_from_model:
    #     H(
    #         "--finetune-from-model",
    #         args.finetune_from_model,
    #         save_dir_key=lambda _: args.model_size,
    #     )

    grid += [
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--no-best-checkpoints"),
        hyperparam("--save-interval", args.save_interval_updates),
        hyperparam("--save-interval-updates", args.save_interval_updates),
        hyperparam("--keep-interval-updates", args.keep_interval_updates),

        hyperparam("--validate-interval-updates", args.interval),
    ]
    # hyperparam("--no-save-optimizer-state"),
    if args.validate_at_beginning:
        grid += [hyperparam("--validate-at-beginning")]
    if args.no_save:
        H("--no-save")
    else:
        H("--best-checkpoint-metric", "loss")

    if args.label_loss:
        H("--task", "streaming_instruction_finetune_language_modeling")
    else:
        H("--task", "streaming_finetune_language_modeling")
    H("--vocab-filename", os.path.join("/gscratch/zlab/sg01/opt/vocab/gpt2-vocab.json"))# , save_dir_key=lambda _: "gpt2")
    H(
        "--merges-filename",
        os.path.join("/gscratch/zlab/sg01/opt/vocab/gpt2-merges.txt"),
    )
    H("--sample-break-mode", args.sbm) # , save_dir_key=lambda val: f"sbm_{val}")

    if args.valid_subset == "valid":
        H(
            "--combine-valid-subsets"
        )  # this by default assumes the split name as valid (in metaseq/main)
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
    # H("--tensor-parallel-init-model-on-gpu")
    # H("--model-parallel-size", size.model_parallel)
    H("--criterion", "moe_cross_entropy")
    # H("--distribute-checkpointed-activations")

    H("--arch", "transformer_lm_gpt3_xl")
    H("--activation-fn", "relu")
    H("--decoder-learned-pos")
    H("--share-decoder-input-output-embed")

    # MOE Hyperparams
    H("--moe-expert-count", args.moe_num_experts, save_dir_key=lambda val: f"nexperts_{val}")

    H("--moe-freq", 2)
    H("--moe-gating-use-fp32")
    if args.moe_initialize_from_opt:
        H("--moe-initialize-from-opt", save_dir_key=lambda val: f"init_opt")
    H("--moe-second-expert-policy", "all")
    H("--moe-normalize-expert-grad", "sqrt_world_size")
    H("--moe-eval-capacity-token-fraction", -1.0)
    H("--moe-gate-loss-wt", 0.01)
    H("--moe-gate-loss-combine-method", "sum")


    # if not args.embdr:
        # H("--no-emb-dropout", save_dir_key=lambda _: "0edr")
    if args.min_loss_scale > 0:
        H("--min-loss-scale", args.min_loss_scale)
    # Add document attention seperator to efficiently finetune under streaming setting.
    if args.self_attn_doc_sep:
        H("--self-attn-doc-sep", 2, save_dir_key=lambda val: f"docsep_{val}")
    H("--checkpoint-activations", binary_flag=True) #, save_dir_key=lambda _: "ckpt")
    # this model requires checkpoint activations to load
    # H("--use-sharded-state")
    H("--decoder-learned-pos")
    # H("--gradient-predivide-factor", 32.0)
    H("--no-scale-embedding")

    H("--tokens-per-sample", args.tps) #, save_dir_key=lambda val: f"tps_{val}")
    H("--ddp-backend", "fully_sharded")
    H("--save-async")
    # H("--quiet")

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
        hyperparam("--batch-size-valid", 8),
        hyperparam("--update-freq", args.uf, save_dir_key=lambda val: f"uf{val}"),
    ]

    # regularization
    dropout = args.dropout
    grid += [
        hyperparam("--dropout", dropout), #, save_dir_key=lambda val: f"dr{val}"),
        # --attention-dropout will be set to mirror --dropout in postprocess_args
        hyperparam(
            "--attention-dropout", dropout), #, save_dir_key=lambda val: f"atdr{val}"),
    ]
    if args.wd > 0:
        H("--weight-decay", args.wd, save_dir_key=lambda val: f"wd{val}")
    is_175B = args.model_size == "175b"
    H("--adam-betas", "(0.9, 0.95)")
    H("--adam-eps", 1e-6)
    H("--clip-norm", args.clip_norm, save_dir_key=lambda val: f"clip{val}" if args.clip_norm < 1.0 else "")
    if not args.no_fp16_adam:
        H("--fp16-adam-stats")
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

    # Below settings are not needed if using `finetune-from-model` args
    # If restore-file is set, then anyway we don't need the reset of meters
    # such that we can continue training

    H("--reset-meters")
    H("--reset-dataloader")
    #H("--reset-optimizer")
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
    args.azure_folder_auto_name = True
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
    parser.add_argument("--moe-initialize-from-opt", action='store_true')
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