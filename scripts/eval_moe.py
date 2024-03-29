import argparse
import os
import subprocess

LEARNING_RATES = {
    "1.3b": 2e-5,
    "6.7b": 1.2e-5
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str)
    parser.add_argument('--data-dir', type=str, default='/gscratch/zlab/sg01/data/c4/')
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--run", choices=['slurm', 'local'])
    parser.add_argument("--partition", type=str, default='ckpt')
    parser.add_argument("--constraint", type=str, default='[rtx6k|a40|a100]')
    parser.add_argument("--checkpoint-prefix", type=str, default='checkpoint_last')
    parser.add_argument("--account", type=str, default=None) 
    parser.add_argument("--job-folder", type=str, default='/gscratch/zlab/sg01/submitit_evals/') 

    args = parser.parse_args()

    job_folder = os.path.join(args.job_folder, os.path.basename(os.path.normpath(args.model_dir)))

    command = f"bash scripts/eval_lm.sh \
                {args.model_dir} \
                {args.num_gpus} \
                {args.num_nodes} \
                {args.num_experts} \
                {args.run} \
                {args.checkpoint_prefix} \
                {args.partition} \
                {args.constraint} \
                {args.data_dir} \
                {args.account} \
                {job_folder} \
                "
    subprocess.run(command.split(), check=True, text=True)