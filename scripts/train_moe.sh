OPT_INIT=$1

if [ $OPT_INIT == "init_opt" ]; then 
    python -m  fairseq_cli.train \
        /gscratch/zlab/sg01/data/c4/  \
        --save-dir /gscratch/zlab/sg01/test_eval_4_exp/ \
        --ddp-backend fully_sharded \
        --memory-efficient-fp16 \
        --checkpoint-activations   \
        --task streaming_finetune_language_modeling \
        --tokens-per-sample 2048   \
        --arch transformer_lm_gpt3_xl \
        --share-decoder-input-output-embed   \
        --moe-expert-count 4 \
        --moe-freq 2   \
        --moe-gating-use-fp32 \
        --moe-second-expert-policy all   \
        --moe-normalize-expert-grad sqrt_world_size   \
        --moe-eval-capacity-token-fraction -1.0  \
        --max-sentences-valid 1 \
        --num-workers-valid 0   \
        --criterion moe_cross_entropy \
        --moe-gate-loss-wt 0.01 \
        --moe-gate-loss-combine-method sum   \
        --optimizer adam \
        --save-interval-updates 1000 \
        --lr-scheduler polynomial_decay \
        --fp16-adam-stats \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.0   \
        --lr 2e-5 \
        --batch-size 8 \
        --update-freq 1   \
        --total-num-update 10000 \
        --max-update 10000 \
        --disable-validation   \
        --log-format json \
        --log-interval 10 \
        --train-subset train \
        --merges-filename /gscratch/zlab/sg01/opt/vocab/gpt2-merges.txt \
        --vocab-filename /gscratch/zlab/sg01/opt/vocab/gpt2-vocab.json \
        --reset-dataloader
else
    python -m  fairseq_cli.train \
        /gscratch/zlab/sg01/data/c4/  \
        --save-dir /gscratch/zlab/sg01/opt_ft/moe_from_scratch/ \
        --ddp-backend fully_sharded \
        --memory-efficient-fp16 \
        --checkpoint-activations   \
        --task streaming_finetune_language_modeling \
        --tokens-per-sample 2048   \
        --arch transformer_lm_gpt3_xl \
        --share-decoder-input-output-embed   \
        --moe-expert-count 2 \
        --moe-freq 2   \
        --moe-gating-use-fp32 \
        --moe-second-expert-policy all   \
        --moe-normalize-expert-grad sqrt_world_size   \
        --moe-eval-capacity-token-fraction -1.0  \
        --max-sentences-valid 1 \
        --num-workers-valid 0   \
        --criterion moe_cross_entropy \
        --moe-gate-loss-wt 0.01 \
        --moe-gate-loss-combine-method sum   \
        --optimizer adam \
        --fp16-adam-stats \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.0   \
        --lr-scheduler polynomial_decay \
        --total-num-update 10000 \
        --warmup-updates 800 \
        --lr 2e-5 \
        --batch-size 8 \
        --update-freq 32   \
        --max-update 10000 \
        --disable-validation   \
        --log-format json \
        --log-interval 10 \
        --train-subset train \
        --merges-filename /gscratch/zlab/sg01/opt/vocab/gpt2-merges.txt \
        --vocab-filename /gscratch/zlab/sg01/opt/vocab/gpt2-vocab.json;
fi;