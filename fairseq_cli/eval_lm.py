#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import math
import os
import submitit
import sys
import time
from argparse import Namespace
from textwrap import indent
from typing import Iterable, List, Optional
from tqdm.auto import tqdm

import torch
import fairseq
from fairseq.file_io import save_json
from fairseq.utils import round_safe
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter
from fairseq.sequence_scorer import SequenceScorer
from omegaconf import DictConfig


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.eval_lm")


def eval_lm(
    models: List[fairseq.models.FairseqModel],
    source_dictionary: fairseq.data.Dictionary,
    batch_iterator: Iterable,
    post_process: Optional[str] = None,
    output_word_probs: bool = False,
    output_word_stats: bool = False,
    target_dictionary: Optional[fairseq.data.Dictionary] = None,
    softmax_batch: int = False,
    remove_bos_token: bool = False,
    device: Optional[torch.device] = None,
    max_valid_steps=None,
):
    """
    Args:
        models (List[~fairseq.models.FairseqModel]): list of models to
            evaluate. Models are essentially `nn.Module` instances, but
            must be compatible with fairseq's `SequenceScorer`.
        source_dictionary (~fairseq.data.Dictionary): dictionary for
            applying any relevant post processing or outputing word
            probs/stats.
        batch_iterator (Iterable): yield batches of data
        post_process (Optional[str]): post-process text by removing BPE,
            letter segmentation, etc. Valid options can be found in
            fairseq.data.utils.post_process, although not all options
            are implemented here.
        output_word_probs (Optional[bool]): output words and their
            predicted log probabilities
        output_word_stats (Optional[bool]): output word statistics such
            as word count and average probability
        target_dictionary (Optional[~fairseq.data.Dictionary]): output
            dictionary (defaults to *source_dictionary*)
        softmax_batch (Optional[bool]): if BxT is more than this, will
            batch the softmax over vocab to this amount of tokens, in
            order to fit into GPU memory
        remove_bos_token (Optional[bool]): if True, confirm that the
            first token is the beginning-of-sentence symbol (according
            to the relevant dictionary) and remove it from the output
        device (Optional[torch.device]): device to use for evaluation
            (defaults to device of first model parameter)
    """
    start_time =time.time()
    if target_dictionary is None:
        target_dictionary = source_dictionary
    if device is None:
        device = next(models[0].parameters()).device

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(target_dictionary, softmax_batch)

    score_sum = 0.0
    count = 0

    if post_process is not None:
        if post_process in {"subword_nmt", "@@ "}:
            bpe_cont = post_process.rstrip()
            bpe_toks = {
                i
                for i in range(len(source_dictionary))
                if source_dictionary[i].endswith(bpe_cont)
            }
        else:
            raise NotImplementedError(
                "--post-process={post_process} is not implemented"
            )
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0

    word_stats = dict()
    first_batch = None
    pbar = tqdm(enumerate(batch_iterator))
    for i, sample in pbar:
        if max_valid_steps is not None and i > max_valid_steps:
            break
        is_dummy_batch = False
        if not first_batch and "net_input" in sample:
            first_batch = sample
        if "net_input" not in sample:
            if first_batch:
                logger.warning("Adding a dummy batch")
                sample = first_batch
                is_dummy_batch = True
            else:
                continue

        sample = utils.move_to_cuda(sample, device=device)

        gen_timer.start()
        hypos = scorer.generate(models, sample)
        gen_timer.stop(sample["ntokens"])

        # Don't calculate score for dummy batch
        if is_dummy_batch:
            continue

        for i, hypos_i in enumerate(hypos):
            hypo = hypos_i[0]
            # sample_id = sample["id"][i]

            tokens = hypo["tokens"]
            tgt_len = tokens.numel()
            pos_scores = hypo["positional_scores"].float()

            if remove_bos_token:
                assert hypo["tokens"][0].item() == target_dictionary.bos()
                tokens = tokens[1:]
                pos_scores = pos_scores[1:]

            skipped_toks = 0
            if bpe_toks is not None:
                for i in range(tgt_len - 1):
                    if tokens[i].item() in bpe_toks:
                        skipped_toks += 1
                        pos_scores[i + 1] += pos_scores[i]
                        pos_scores[i] = 0

            inf_scores = pos_scores.eq(float("inf")) | pos_scores.eq(float("-inf"))
            if inf_scores.any():
                logger.info(
                    "skipping tokens with inf scores:",
                    target_dictionary.string(tokens[inf_scores.nonzero()]),
                )
                pos_scores = pos_scores[(~inf_scores).nonzero()]
            score_sum += pos_scores.sum().cpu()
            count += pos_scores.numel() - skipped_toks
            running_loss = get_aggregated_loss(score_sum, count)
            running_ppl = 2 ** running_loss
            pbar.set_description(f'ppl: {running_ppl}')

    avg_nll_loss = get_aggregated_loss(score_sum, count)  # convert to base 2
    tokens, gpu_seconds_taken, avg_time = get_aggregated_timer_stats(gen_timer)
    end_time = time.time()
    tot_time = end_time-start_time
    logger.info(f"Evaluated {tokens:,} tokens in {tot_time:.1f}s ({tokens / tot_time:.2f} tokens/s)")
    if output_word_stats:
        for ws in sorted(word_stats.values(), key=lambda x: x.count, reverse=True):
            logger.info(ws)

    return {
        "loss": avg_nll_loss,
        "perplexity": 2 ** avg_nll_loss,
        "r0_tps_step": 1.0 / gen_timer.avg if gen_timer.avg > 0 else 0,
        "ntok_total": tokens,
        "gpu_step_seconds": gpu_seconds_taken,
    }


def _all_reduce_float(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    x_tensor = x.cuda()
    torch.distributed.all_reduce(x_tensor)
    return x_tensor.item()


def get_aggregated_loss(score_sum, count):
    if torch.distributed.is_initialized():
        # logger.warning("Aggregating scores across the distributed world")
        count = _all_reduce_float(count)
        score_sum = _all_reduce_float(score_sum)
    return (
        -score_sum / count / math.log(2) if count > 0 else 0
    )


def get_aggregated_timer_stats(gen_timer):
    tokens, time_taken, avg_time = gen_timer.n, gen_timer.sum, 1.0 / gen_timer.avg if gen_timer.avg > 0 else 0
    if torch.distributed.is_initialized():
        logger.warning("Aggregating timer stats across the distributed world")
        tokens = _all_reduce_float(tokens)
        time_taken = _all_reduce_float(time_taken)
        avg_time = _all_reduce_float(avg_time) / torch.distributed.get_world_size()
    return tokens, time_taken, avg_time


class WordStat(object):
    def __init__(self, word, is_bpe):
        self.word = word
        self.is_bpe = is_bpe
        self.log_prob = 0
        self.next_word_prob = 0
        self.count = 0
        self.missing_next_words = 0

    def add(self, log_prob, next_word_prob):
        """increments counters for the sum of log probs of current word and next
        word (given context ending at current word). Since the next word might be at the end of the example,
        or it might be not counted because it is not an ending subword unit,
        also keeps track of how many of those we have seen"""
        if next_word_prob is not None:
            self.next_word_prob += next_word_prob
        else:
            self.missing_next_words += 1
        self.log_prob += log_prob
        self.count += 1

    def __str__(self):
        return "{}\t{}\t{}\t{}\t{}\t{}".format(
            self.word,
            self.count,
            self.log_prob,
            self.is_bpe,
            self.next_word_prob,
            self.count - self.missing_next_words,
        )


def eval_dataset(cfg: DictConfig, eval_split, task, models, start_time):
    dataset = task.dataset(eval_split)
    # logger.info(f"{cfg.task.data} {eval_split} {len(dataset):,} examples")
    num_shards = max(
        cfg.dataset.num_shards,
        cfg.distributed_training.distributed_world_size,
    )
    shard_id = max(
        cfg.dataset.shard_id,
        cfg.distributed_training.distributed_rank,
    )

    itr = task.get_batch_iterator(
            dataset=task.dataset(eval_split),
            max_tokens=cfg.dataset.max_tokens_valid,
            max_sentences=cfg.dataset.batch_size_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[model.max_positions() for model in models],
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=1,
            shard_id=0,
            num_workers=cfg.dataset.num_workers_valid,
            # always pass a fixed "epoch" to keep validation data consistent
            # across training epochs
            epoch=1,
            data_buffer_size=cfg.dataset.data_buffer_size,
            disable_iterator_cache=True,
            skip_remainder_batch=False,
        ).next_epoch_itr(
                shuffle=False, set_dataset_epoch=False  # use a fixed valid set
                )

    itr = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )
    load_time = time.time() - start_time
    logger.info(f'load time: {load_time:.2f} seconds')
    results = eval_lm(
        models=models,
        source_dictionary=task.source_dictionary,
        batch_iterator=itr,
        post_process=cfg.common_eval.post_process,
        output_word_probs=cfg.eval_lm.output_word_probs,
        output_word_stats=cfg.eval_lm.output_word_stats,
        target_dictionary=task.target_dictionary,
        softmax_batch=cfg.eval_lm.softmax_batch,
        remove_bos_token=getattr(cfg.task, "add_bos_token", False),
        max_valid_steps=cfg.eval_lm.max_valid_steps,
    )

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(
        "{} Loss (base 2): {:.4f}, Perplexity: {:.2f}".format(
            eval_split, results["loss"], results["perplexity"]
        )
    )

    if isinstance(cfg.eval_lm.stats_path, str):
        rr = {k: round_safe(v) for k,v in results.items()}
        rr['wall_time'] = round_safe(total_time)
        rr['wall_time_load'] = round_safe(load_time)
        rr['wall_time_model'] = round_safe(total_time - load_time)
    else:
        rr = None

    return results, rr, end_time

def main(cfg: DictConfig, **unused_kwargs):
    start_time = time.time()
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    logger.info('---------------------------')
    logger.info('cfg:')
    config_content = getattr(cfg, "_content")
    for config_key in config_content:
        logger.info(config_key + '\t' + str(config_content[config_key]))
    logger.info('---------------------------')

    if cfg.eval_lm.context_window > 0:
        # reduce tokens per sample by the required context window size
        cfg.task.tokens_per_sample -= cfg.eval_lm.context_window

    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    model_overrides = eval(cfg.common_eval.model_overrides)
    is_base_moe = model_overrides.get('is_base_moe', False)
    if cfg.common_eval.is_moe or is_base_moe:
        rank = distributed_utils.get_data_parallel_rank()
        cfg.checkpoint.checkpoint_suffix = f"-rank-{rank}-shard{rank}"
        is_moe = True
        # This is required for making all_to_all work on same sized tensors across gpus.
        # cfg['task']['pad_to_fixed_length'] = True
    else:
        is_moe = False

    # Initialize the task using the current *cfg*
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    model_overrides['batch_size_valid'] = cfg.dataset.batch_size
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=model_overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
        task=task,
        is_moe=is_moe or is_base_moe,
    )

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    # Optimize ensemble for generation and set the source and dest dicts on the model
    # (required by scorer)
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()

        if is_moe:
            # For moe models, we want to enable padding in moe layer, so not calling this.
            model.prepare_for_inference_(cfg)

    assert len(models) > 0

    logger.info(
        "num. model params: {:,}".format(sum(p.numel() for p in models[0].parameters()))
    )

    # Load dataset splits
    task.load_dataset(cfg.dataset.gen_subset)
    eval_splits = [cfg.dataset.gen_subset]
    if cfg.task._name == 'multilingual_language_modeling':
        languages = cfg.task.langs.split(',')
        for lang in languages:
            eval_splits.append(f'{cfg.dataset.gen_subset}_{lang}')
    
    all_split_results = dict()
    for eval_split in eval_splits:
        results, rr, end_time = eval_dataset(cfg, eval_split, task, models, start_time)
        start_time = end_time
        all_split_results[eval_split] = rr

    if isinstance(cfg.eval_lm.stats_path, str):
        save_path = f'{cfg.eval_lm.stats_path}.json'
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            save_json(all_split_results, save_path)
            logger.info('Evaluation results saved to {}'.format(save_path))

    return results


from fairseq import distributed_utils

def call_main(cfg, main, **kwargs):
    cfg.distributed_training.distributed_port = 56000
    if cfg.distributed_training.distributed_init_method is None:
        distributed_utils.infer_init_method(cfg.distributed_training)

    if cfg.distributed_training.distributed_init_method is not None:
        # distributed training
        if not cfg.distributed_training.distributed_no_spawn:
            start_rank = cfg.distributed_training.distributed_rank
            cfg.distributed_training.distributed_rank = None  # assign automatically
            kwargs["start_rank"] = start_rank
            return distributed_utils._spawn_helper(main, cfg, kwargs)
        else:
            return distributed_utils.distributed_main(
                cfg.distributed_training.device_id, main, cfg, kwargs
            )
    else:
        # single GPU main
        return main(cfg, **kwargs)


def runner(x):
    return call_main(x, main)

def cli_main():
    parser = options.get_eval_lm_parser()
    parser.add_argument("--submitit", action='store_true')
    parser.add_argument("--partition", type=str)
    parser.add_argument("--constraint", type=str)
    parser.add_argument("--account", type=str)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--job-folder", type=str)
    args = options.parse_args_and_arch(parser)

    if args.submitit:
        executor = submitit.AutoExecutor(folder=args.job_folder, slurm_max_num_timeout=30)
        num_gpus_per_node = args.num_gpus
        nodes = args.num_nodes
        timeout_min = 60

        kwargs = {}

        executor.update_parameters(
                mem_gb=40 * num_gpus_per_node,
                gpus_per_node=num_gpus_per_node,
                tasks_per_node=num_gpus_per_node,  # one task per GPU
                cpus_per_task=2,
                nodes=nodes,
                timeout_min=timeout_min,  # max is 60 * 72
                # Below are cluster dependent parameters
                slurm_partition=args.partition,
                slurm_constraint=args.constraint,
                **kwargs
            )
        if args.account and args.account != "None":
            executor.update_parameters(
                slurm_account=args.account,
            )

        executor.update_parameters(name="eval")

        func = lambda x: call_main(x, main)

        executor.submit(func, convert_namespace_to_omegaconf(args))
        logger.info(f"launched {num_gpus_per_node * nodes} GPU evaluation job via submitit")
    else:
        distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)
    

if __name__ == "__main__":
    cli_main()