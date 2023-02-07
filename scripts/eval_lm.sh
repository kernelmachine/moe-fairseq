
MODEL_DIR=$1
NUM_EXPERTS=$2
SLURM=$3
CHECKPOINT_TO_PROCESS=$4
PARTITION=$5
CONSTRAINT=$6

echo $CHECKPOINT_TO_PROCESS

TOKENS_PER_SAMPLE=2048
BATCH_SIZE=1
MODEL_CAPACITY=$(( 2 * (8  * $TOKENS_PER_SAMPLE) / ${NUM_EXPERTS} )) # based on train script = 2 * (local_batch_size)/(global_num_experts) = 2 * (8*1024)/512 = 32



MOE_EVAL_CAPACITY_TOKEN_FRACTION=`python3 -c "print($BATCH_SIZE * $TOKENS_PER_SAMPLE/$MODEL_CAPACITY)"` 
DATA_PATH=/gscratch/zlab/sg01/data/c4/

if [ $SLURM == "slurm" ]; then
  SUBMITIT_PHRASE="--submitit"
else
  SUBMITIT_PHRASE="";
fi;


# echo $MODEL_DIR
# # create temporary model checkpoint directory and create symlinks
# RANK_PATHS=`find $MODEL_DIR -name $CHECKPOINT_TO_PROCESS-rank-*.pt`
# echo $RANK_PATHS
# TEMP_FOLDER=`mktemp -d`
# pushd $TEMP_FOLDER
# for m in $RANK_PATHS;
# do
#     filename=`echo $m | rev | cut -d '/' -f1 | rev | sed 's/-shard[0-9]*//g'` # extract only filename from full path
#     ln -s $m ./$filename 
# done;
# SHARED_PATH=`find $MODEL_DIR -name $CHECKPOINT_TO_PROCESS-shared-shard0.pt`
# filename=`echo $SHARED_PATH | rev | cut -d '/' -f1 | rev | sed 's/-shard[0-9]*//g'`
# ln -s $SHARED_PATH ./$filename
# popd

# ls $TEMP_FOLDER

# set -ux
python -m fairseq_cli.eval_lm \
  $DATA_PATH \
  --ddp-backend c10d \
  --path $MODEL_DIR/$CHECKPOINT_TO_PROCESS.pt \
  --task streaming_finetune_language_modeling \
  --gen-subset valid_c4_small/C4_small \
  --sample-break-mode eos_pad_8 \
  --merges-filename /gscratch/zlab/sg01/opt/vocab/gpt2-merges.txt \
  --vocab-filename /gscratch/zlab/sg01/opt/vocab/gpt2-vocab.json \
  --tokens-per-sample $TOKENS_PER_SAMPLE \
  --batch-size $BATCH_SIZE \
  --max-valid-steps 200 \
  --fp16  --is-moe --distributed-world-size $NUM_EXPERTS \
  --model-overrides "{'world_size': $NUM_EXPERTS, 'moe_eval_capacity_token_fraction': $MOE_EVAL_CAPACITY_TOKEN_FRACTION}" \
  --log-format json \
  --partition $PARTITION \
  --constraint $CONSTRAINT \
  --num-experts $NUM_EXPERTS \
  $SUBMITIT_PHRASE
