# python -m  fairseq_cli.train \
#     /gscratch/zlab/sg01/data/c4/  \
#     --save-dir /gscratch/zlab/sg01/opt_ft/test_moe/ \
#     --ddp-backend fully_sharded \
#     --memory-efficient-fp16 \
#     --checkpoint-activations   \
#     --task streaming_finetune_language_modeling \
#     --tokens-per-sample 2048   \
#     --arch transformer_lm_gpt \
#     --share-decoder-input-output-embed   \
#     --decoder-layers 24 \
#     --decoder-embed-dim 2048 \
#     --decoder-ffn-embed-dim 8192   \
#     --decoder-attention-heads 32   \
#     --moe-expert-count 4 \
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
#     --fp16-adam-stats \
#     --adam-betas '(0.9, 0.98)' \
#     --clip-norm 0.0   \
#     --lr 0.00002 \
#     --batch-size 4 \
#     --update-freq 1   \
#     --max-update 10 \
#     --disable-validation   \
#     --log-format json \
#     --log-interval 10 \
#     --train-subset train \
#     --merges-filename /gscratch/zlab/sg01/opt/vocab/gpt2-merges.txt \
#     --vocab-filename /gscratch/zlab/sg01/opt/vocab/gpt2-vocab.json 



python -m  fairseq_cli.train \
    /gscratch/zlab/sg01/data/c4/  \
    --save-dir /gscratch/zlab/sg01/opt_ft/moe/ \
    --ddp-backend fully_sharded \
    --memory-efficient-fp16 \
    --checkpoint-activations   \
    --task streaming_finetune_language_modeling \
    --tokens-per-sample 2048   \
    --arch transformer_lm_gpt \
    --share-decoder-input-output-embed   \
    --decoder-layers 24 \
    --decoder-embed-dim 2048 \
    --decoder-ffn-embed-dim 8192   \
    --decoder-attention-heads 32   \
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
    --fp16-adam-stats \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0   \
    --lr 0.00002 \
    --batch-size 4 \
    --update-freq 1   \
    --max-update 10000 \
    --disable-validation   \
    --log-format json \
    --log-interval 10 \
    --train-subset train \
    --merges-filename /gscratch/zlab/sg01/opt/vocab/gpt2-merges.txt \
    --vocab-filename /gscratch/zlab/sg01/opt/vocab/gpt2-vocab.json \
    --finetune-from-opt \
    --reset-optimizer \
    --reset-dataloader \
    --reset-lr-scheduler \
    --reset-meters
