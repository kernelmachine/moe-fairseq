# Training MoE language models

## Setup
https://github.com/kernelmachine/cbtm/tree/main

### Install fairseq
```
git clone 
cd fairseq/
pip install --editable .
```


## Single-node training

The following command will benchmark an MoE language model using synthetic data
on 8 GPUs. The model has 8 experts (one per GPU) and 4.1B parameters total.

```bash
# set NUM_EXPERTS based on # of GPUs and desired # experts per GPU
# generally it's recommended to have a single expert per GPU
python -m scripts.train_moe \
--initialization opt \
--model-size 1.3b \
--run slurm \
--data c4 \
--num-nodes 2 \
--num-gpus 8 \
--num-experts 32 \
--max-steps 10000 \
--lr 2e-5 \
--update-freq 4 \
--batch-size 2 \
--partition learnlab \
--constraint 'volta32gb' # set this to volta32gb
```

# Evaluating MoE language models

After you train, 
```bash
python -m scripts.eval_moe \
--model-dir /checkpoint/margaretli/opt_ft/moe/finetune.moe.opt_data.nexperts_32.init_opt.0edr.mu10000.wu0.bsz2.uf4.fp16adam.rs1234.lr2e-05.pat_10000.ngpu8/ \
--data-dir /checkpoint/margaretli/data/opt/120321/ \
--num-experts 32 \
--num-gpus 8 \
--num-nodes 1 \
--run slurm \
--checkpoint-prefix checkpoint_last \
--partition scavenge \
--constraint 'volta32gb' \
--job-folder '/private/home/margaretli/submitit_evals' 
```
