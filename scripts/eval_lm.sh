NUM_GPUS=2

MODEL_DIR='/gscratch/zlab/sg01/test_eval_2_exp/'

CHECKPOINT_TO_PROCESS='checkpoint_last'

TOKENS_PER_SAMPLE=2048
BATCH_SIZE=1
MODEL_CAPACITY=32 # based on train script = 2 * (local_batch_size)/(global_num_experts) = 2 * (8*1024)/512 = 32
MOE_EVAL_CAPACITY_TOKEN_FRACTION=`python3 -c "print($MODEL_CAPACITY/($BATCH_SIZE * $TOKENS_PER_SAMPLE))"` 
DATA_PATH=/gscratch/zlab/sg01/data/c4/

# create temporary model checkpoint directory and create symlinks
RANK_PATHS=`find $MODEL_DIR -name $CHECKPOINT_TO_PROCESS-rank-*.pt`
TEMP_FOLDER=`mktemp -d`
pushd $TEMP_FOLDER
for m in $RANK_PATHS;
do
    filename=`echo $m | rev | cut -d '/' -f1 | rev | sed 's/-shard[0-9]*//g'` # extract only filename from full path
    ln -s $m ./$filename 
done;
SHARED_PATH=`find $MODEL_DIR -name $CHECKPOINT_TO_PROCESS-shared-shard0.pt`
filename=`echo $SHARED_PATH | rev | cut -d '/' -f1 | rev | sed 's/-shard[0-9]*//g'`
ln -s $SHARED_PATH ./$filename
popd

set -ux
CUDA_VISIBLE_DEVICES=0,1 python -m fairseq_cli.eval_lm \
  $DATA_PATH \
  --ddp-backend c10d \
  --path $TEMP_FOLDER/$CHECKPOINT_TO_PROCESS.pt \
  --task streaming_finetune_language_modeling \
  --gen-subset valid_c4_small/C4_small \
  --sample-break-mode none \
  --merges-filename /gscratch/zlab/sg01/opt/vocab/gpt2-merges.txt \
  --vocab-filename /gscratch/zlab/sg01/opt/vocab/gpt2-vocab.json \
  --tokens-per-sample $TOKENS_PER_SAMPLE \
  --batch-size $BATCH_SIZE \
  --fp16  --is-moe --distributed-world-size $NUM_GPUS \
  --model-overrides "{'world_size': $NUM_GPUS, 'moe_eval_capacity_token_fraction': 1.0}" \
  --log-format json