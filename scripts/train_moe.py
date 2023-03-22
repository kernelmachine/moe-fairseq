import argparse

import subprocess

LEARNING_RATES = {
    "1.3b": 2e-5,
    "6.7b": 1.2e-5
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialization", choices=['opt', 'random'])
    parser.add_argument("--model-size", choices=['6.7b','1.3b', 'resharded_1.3b', '125m', '350m'])
    parser.add_argument("--run", choices=['slurm', 'local'])
    parser.add_argument("--data", choices=['imdb','c4', 'opt_data', 's2orc_data', 'pile_data_alt', 'flan_data', 'demix_data', 'stories', 'Wikipedia_en', 'DM_Mathematics', 'OpenWebText2', 'Gutenberg_PG-19', 'redditflattened', 'HackerNews', 'CommonCrawl', 'BookCorpusFair', 'Enron_Emails', 'ccnewsv2', 'USPTO', 'OpenSubtitles'])
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--fastforward-to-epoch", type=int, default=None)

    parser.add_argument("--partition", type=str)
    parser.add_argument("--constraint", type=str, default='[rtx6k|a40|a100]')

    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--update-freq", "-uf", type=int, default=1)
    parser.add_argument("--batch-size", "-bs", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=32)

    args = parser.parse_args()
    
    world_size = args.num_gpus * args.num_nodes
    learning_rate = args.lr or LEARNING_RATES[args.model_size]
    # TODO(suchin): removed this
    # end_lr = learning_rate * 0.1
    end_lr = 0.0
    command = f"bash /private/home/margaretli/gitfiles/moe-fairseq/scripts/train_moe.sh \
                {args.initialization} \
                {args.lr} \
                {args.num_nodes} \
                {args.num_gpus} \
                {args.model_size} \
                {args.run} \
                {args.data} \
                {args.max_steps} \
                {args.update_freq} \
                {args.partition} \
                {args.constraint} \
                {end_lr} \
                {args.fastforward_to_epoch} \
                {args.batch_size} \
                {args.num_experts} \
                "
    subprocess.run(command.split(), check=True, text=True)
