OPT_INIT=$1
LR=$2
NUM_NODES=$3
NUM_GPUS=$4
MODEL_SIZE=$5
SLURM=$6
DATA=$7
MAX_STEPS=$8
UPDATE_FREQ=${9}
PARTITION=${10}
CONSTRAINT=${11}
END_LR=${12}
FASTFORWARD=${13}
BATCH_SIZE=${14}
NUM_EXPERTS=${15}
# TODO(margaret): change this
CHECKPOINT_DIR="/checkpoint/$USER/opt_ft/moe/";
PROJECT="finetune.moe.$DATA";



if [ $OPT_INIT == "opt" ]; then 
    INIT_PHRASE="--moe-initialize-from-opt /checkpoint/rpasunuru/opt_models/1.3B_gptz/consolidated.pt"
else
    INIT_PHRASE="";
fi;

if [ $SLURM == "slurm" ]; then
    LOCAL_PHRASE="";
else
    LOCAL_PHRASE="--dry-run --local";
fi;

JOBARRAY_NAME="moe";

if [ $SLURM == "slurm" ]; then
    JOBARRAY_PHRASE="--use-jobarray --jobarray-name $JOBARRAY_NAME ";
else
    JOBARRAY_PHRASE="";
fi;


if [ $FASTFORWARD != "None" ]; then
    FASTFORWARD_PHRASE="--fast-forward $FASTFORWARD ";
else
    FASTFORWARD_PHRASE="";
fi;


# reducing batch size to 2
# if [ $PARTITION != "gpu-a40" ]; then
#     BS=2;
#     UF=$(( ${UPDATE_FREQ}  * 4 ));
# else
BS=${BATCH_SIZE};
UF=${UPDATE_FREQ};
# fi;

#NUM_EXPERTS=$(( ${NUM_GPUS}  * ${NUM_NODES} ));
# NUM_EXPERTS=32;

python -m fairseq.fb_sweep.ft_stream \
    -n $NUM_NODES \
    -g $NUM_GPUS \
    -p $PROJECT \
    --checkpoints-dir $CHECKPOINT_DIR \
    --model-size $MODEL_SIZE \
    --data-type $DATA \
    --lr $LR \
    --wd 0.0 \
    --moe-num-experts $NUM_EXPERTS \
    --warmup-update 0 \
    --max-update ${MAX_STEPS} \
    --uf $UF \
    --no-tensorboard \
    --no-wandb \
    --interval 1000 \
    --save-interval-updates 250 \
    --keep-interval-updates 2 \
    --bs $BS \
    --sbm none \
    --partition $PARTITION \
    --constraint $CONSTRAINT \
    --resume-failed \
    $FASTFORWARD_PHRASE \
    --dropout 0.1 \
    --end-learning-rate $END_LR \
    --fair \
    --max-valid-steps 100 \
    $INIT_PHRASE \
    $LOCAL_PHRASE \
    $JOBARRAY_PHRASE \
    --script /private/home/$USER/gitfiles/moe-fairseq/fairseq_cli/train.py  # TODO(margaret): change this


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
