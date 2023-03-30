INIT_CHECKPOINT=$1
MOE_EXPERT_STATE_DICT=$2
LR=$3
NUM_NODES=$4
NUM_GPUS=$5
MODEL_SIZE=$6
SLURM=$7
DATA_DIR=$8
MAX_STEPS=$8
UPDATE_FREQ=${9}
PARTITION=${10}
CONSTRAINT=${11}
END_LR=${12}
BATCH_SIZE=${14}
NUM_EXPERTS=${15}
# TODO(margaret): change this
CHECKPOINT_DIR="/checkpoint/$USER/opt_ft/moe/";
PROJECT="finetune.moe.$DATA";


JOBARRAY_NAME="moe";

if [ $SLURM == "slurm" ]; then
    LOCAL_PHRASE="";
    JOBARRAY_PHRASE="--use-jobarray --jobarray-name $JOBARRAY_NAME ";
else
    LOCAL_PHRASE="--dry-run --local";
    JOBARRAY_PHRASE="";
fi;


python -m fairseq.fb_sweep.ft_stream \
$DATA_DIR \
    -n $NUM_NODES \
    -g $NUM_GPUS \
    -p $PROJECT \
    --checkpoints-dir $CHECKPOINT_DIR \
    --model-size $MODEL_SIZE \
    --lr $LR \
    --wd 0.0 \
    --moe-initialize-from-opt $INIT_CHECKPOINT \
    --moe-path-to-expert-state-dict $MOE_EXPERT_STATE_DICT \
    --moe-num-experts $NUM_EXPERTS \
    --warmup-update 0 \
    --max-update $MAX_STEPS \
    --uf $UPDATE_FREQ \
    --no-tensorboard \
    --no-wandb \
    --interval 1000 \
    --save-interval-updates 250 \
    --keep-interval-updates 2 \
    --bs $BATCH_SIZE \
    --sbm none \
    --partition $PARTITION \
    --constraint $CONSTRAINT \
    --resume-failed \
    --dropout 0.1 \
    --end-learning-rate $END_LR \
    --fair \
    --max-valid-steps 100 \
    $INIT_PHRASE \
    $LOCAL_PHRASE \
    $JOBARRAY_PHRASE \
    --script $MOE_FAIRSEQ_FOLDER/fairseq_cli/train.py  
