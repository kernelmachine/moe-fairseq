import argparse

import subprocess

LEARNING_RATES = {
    "1.3b": 2e-5,
    "6.7b": 1.2e-5
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--run", choices=['slurm', 'local'])
    parser.add_argument("--partition", type=str)

    args = parser.parse_args()
    
    command = f"bash /gscratch/zlab/sg01/fairseq/scripts/eval_lm.sh \
                {args.model_dir} \
                {args.num_experts} \
                {args.run} \
                {args.partition} \
                "
    subprocess.run(command.split(), check=True, text=True)
