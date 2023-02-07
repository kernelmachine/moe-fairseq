# Training MoE language models

## Dependencies

Follow the installation instructions at our metaseq fork:
https://github.com/kernelmachine/metaseq/blob/main/docs/setup.md

But don't install metaseq, just install fairseq by cd'ing into this directory:

```
cd fairseq/
pip install --editable .
```

## Change cluster/SLURM environment variables

Search for the following string

```
# TODO(margaret): change this
```

for lines to change to FAIR cluster env. I think I marked everything!

## Training MoE language models

Here we do sparse upcycling from an OPT checkpoint. Assume 1 expert per GPU, and topk=2.

```bash
python -m scripts.train_moe \
  --initialization opt \
  --model-size 1.3b \
  --run slurm \
  --data c4 \
  --num-nodes 1 \
  --num-gpus 8 \
  --max-steps 10000 \ # set this to 7869 to FLOP-match our cluster models
  --lr 2e-5 \ # 2e-4 results in loss exploding for all budgets I have tried
  --update-freq 1 \
  --partition ckpt \ # set this to fair cluster partitions
  --constraint '[rtx6k|a40|a100]' # set this to volta32gb
```

# Evaluating MoE language models

This will run eval via submitit.

```bash
python scripts/eval_moe.py \
  --model-dir /gscratch/zlab/sg01/opt_ft/moe/moe/finetune.moe.c4.nexperts_32.init_opt.0edr.mu7869.wu0.bsz2.uf4.fp16adam.rs1234.lr2e-05.pat_10000.ngpu32/ \ # change this to trained model path
  --data-dir /path/to/c4/data/ \ # change this to path to c4 data
  --num-experts 32 \
  --run slurm \
  --checkpoint-prefix checkpoint_last \
  --partition ckpt \ # change this to fair cluster partitions
  --account zlab \ # change this to fair account
  --constraint '[rtx6k|a40|a100]' \ # change this to volta32gb
  --job-folder '/gscratch/zlab/sg01/submitit_evals' \ # change this to wherever you want to output. I just look at the logs of the running job to check perf.
```
