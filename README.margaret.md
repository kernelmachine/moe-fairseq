# Training MoE language models

## Dependencies

### Create a new conda env
```
conda create --name cbtm python=3.9
conda activate cbtm
```
### Install PyTorch
```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
### Install Apex
```
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout e2083df5eb96643c61613b9df48dd4eea6b07690
```
Depending on your hardware, you may need to comment out lines 101-107 in setup.py before running the next pip install.
```
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```
### Install Megatron
```
git clone --branch fairseq_v2 https://github.com/ngoyal2707/Megatron-LM.git
cd Megatron-LM
pip3 install six regex
pip3 install -e .
```
### Install fairscale
```
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale
git checkout prefetch_fsdp_params_simple
pip3 install -e .
```

### Install fairseq
```
git clone #TODO @margaretli
cd fairseq/
pip install --editable .
```

## Change cluster/SLURM environment variables

Search for the following string

```
# TODO(margaret): change this
```

for lines to change to FAIR cluster env. I think I marked everything!

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
python scripts/eval_moe.py \
--model-dir /checkpoint/margaretli/opt_ft/moe/finetune.moe.opt_data.nexperts_32.init_opt.0edr.mu7869.wu0.bsz2.uf4.fp16adam.rs1234.lr2e-05.pat_10000.ngpu32/ \
--data-dir /checkpoint/suching/data/opt/120321/ \
--num-experts 32 \
--num-gpus 8 \
--num-nodes 2 \
--run slurm \
--checkpoint-prefix checkpoint_last \
--partition scavenge \
--constraint 'volta32gb' \
--job-folder '/private/home/margaretli/submitit_evals' 
```
