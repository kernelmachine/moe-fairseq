# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import torch
import numpy as np
from collections import defaultdict, OrderedDict
from glob import glob

from fairseq import distributed_utils
from fairseq.file_io import torch_load_cpu
from typing import List, Dict
from fairscale.nn.data_parallel.fsdp_optim_utils import is_singleton_tensor
from pathlib import Path
from copy import copy
from omegaconf import OmegaConf

OPT_KEY = 'last_optimizer_state'
logger = logging.getLogger(__name__)


def merge_expert_and_shared_state(expert_state, shared_state):
    state = {}
    for key in ['cfg', 'args', 'extra_state', 'optimizer_history']:
        state[key] = expert_state[key]
    state['model'] = {**expert_state['model'], **shared_state['model']}

    if OPT_KEY in expert_state:
        state[OPT_KEY] = {}
        for key in ['loss_scale', 'param_groups']:
            if key in expert_state[OPT_KEY]:
                state[OPT_KEY][key] = expert_state[OPT_KEY][key]

        if 'param_id_map' in shared_state[OPT_KEY]:  # FSDP
            unflat_expert_state = _unflat_expert_tensor_state(expert_state[OPT_KEY], shared_state[OPT_KEY])
            state[OPT_KEY]['state'] = {
                **shared_state[OPT_KEY]['state'],
                **unflat_expert_state
            }

            state[OPT_KEY].update({k: v for k, v in shared_state[OPT_KEY].items()
                                   if k not in state[OPT_KEY]})
        else:
            state[OPT_KEY]['state'] = {
                **expert_state[OPT_KEY]['state'],
                **shared_state[OPT_KEY]['state'],
            }
    return state


def split_shared_and_expert_states(model, optimizer):
    model_state_dict = model.state_dict()
    shared_model_state_dict = OrderedDict()
    expert_model_state_dict = OrderedDict()
    for name, value in model_state_dict.items():
        # TODO: this is a bit hacky - find a better way determine expert params
        if 'expert' in name and 'expert_centroids' not in name:
            expert_model_state_dict[name] = value
        else:
            shared_model_state_dict[name] = value

    shared_optimizer_state_dict = {}
    expert_optimizer_state_dict = {}
    optimizer_state_dict = optimizer.state_dict()
    for key in ['param_groups', 'loss_scale']:
        if key in optimizer_state_dict:
            expert_optimizer_state_dict[key] = optimizer_state_dict[key]
            shared_optimizer_state_dict[key] = optimizer_state_dict[key]

    param_mappings = {}
    param_id_to_is_expert = {}
    start_index = 0
    for group in optimizer.param_groups:
        # nonlocal start_index
        packed = {k: v for k, v in group.items() if k != 'params'}
        for i, p in enumerate(group['params'], start_index):
            if id(p) not in param_mappings:
                param_mappings.update({id(p): i})
                param_id_to_is_expert[i] = hasattr(p, 'expert') or hasattr(p, 'base_expert')
        packed['params'] = [param_mappings[id(p)] for p in group['params']]
        start_index += len(packed['params'])
        # return packed

    # param_groups = [pack_group(g) ]
    expert_optimizer_state_dict['state'] = {
        k: v for k, v in optimizer_state_dict['state'].items()
        if param_id_to_is_expert[k]
    }
    shared_optimizer_state_dict['state'] = {
        k: v for k, v in optimizer_state_dict['state'].items()
        if not param_id_to_is_expert[k]
    }
    return (
        (shared_model_state_dict, shared_optimizer_state_dict),
        (expert_model_state_dict, expert_optimizer_state_dict),
    )


def merge_multi_local_expert_states(expert_states: List[Dict]) -> Dict:
    merged_expert_state = {}
    for key in ['cfg', 'args', 'extra_state', 'optimizer_history']:
        merged_expert_state[key] = expert_states[0][key]

    if OPT_KEY in expert_states[0]:
        logger.warning(
            "Not stitching last optimizer state while merging experts. "
            "This is okay for inference but not for continued training. "
        )

    model_state_dict = {}
    for expert_group_id, expert_state in enumerate(expert_states):
        num_local_experts_in_chkpt = 1
        for key in expert_state['model']:
            match = re.search(r"experts.([1-9][0-9]*)", key)
            if match and int(match.groups()[0]) + 1 > num_local_experts_in_chkpt:
                num_local_experts_in_chkpt = int(match.groups()[0]) + 1
        logger.info(f"found {num_local_experts_in_chkpt} local experts in expert_group_id={expert_group_id}")
        for key, val in expert_state['model'].items():
            match = re.search(r"experts.([0-9][0-9]*)", key)
            assert match is not None, "\"experts.([0-9][0-9]*)\" pattern expected in key {key}"
            local_chkpt_expert_id = int(match.groups()[0])
            target_expert_id = expert_group_id * num_local_experts_in_chkpt + local_chkpt_expert_id
            key = key.replace(f"experts.{local_chkpt_expert_id}", 'experts.{}'.format(target_expert_id))
            model_state_dict[key] = val
    merged_expert_state['model'] = model_state_dict
    return merged_expert_state


def initialize_moe_from_opt(output_dir, num_experts):

    if Path(output_dir).exists() and len(list(Path(output_dir).glob("*.pt"))) > 1:
        logger.info("output directory not empty, skipping OPT initialization...")
        return
    
    # get OPT state
    #     
    # # get an example state_dict from an MOE model
    # # shared_state_dict = torch.load('/gscratch/zlab/sg01/en_moe_lm_15b/model-shared.pt')
    # 

    # 


    # for key in opt_state['model'].keys():

    # expert_state_dict['model'] = {x: y for x,y in expert_state_dict['model'].items() if 'experts.0.' in x}
    # for key in expert_state_dict['model']:
    #     opt_key = re.sub(".moe_layer.experts.\d+", '', key)
        
    #     expert_state_dict['model'][key] = opt_state['model'][opt_key].clone()
    #     _ = opt_state['model'].pop(opt_key)

    # shared_state_dict = opt_state.copy()
    
    # # for key in shared_state_dict['model']:
    #     # shared_state_dict['model'][key] = opt_state[key]

    # expert_state_dict['cfg']['model']['moe_expert_count'] = num_experts
    # # shared_state_dict['cfg']['model']['moe_expert_count'] = num_experts
    
    # from fairseq import pdb; pdb.set_trace()
    # shared_state = {}
    # expert_state = {}
    # opt_state = torch.load('/gscratch/zlab/sg01/opt_ft/dense/1_cluster/finetune.opt.c4.1.3b.0edr.mu10000.wu0.bsz8.uf1.fp16adam.rs1234.lr2e-05.pat_10000.ngpu4/consolidated.pt')
    path_to_opt = '/gscratch/zlab/sg01/opt/1.3b/consolidated.pt'
    # path_to_opt = '/gscratch/zlab/sg01/opt_ft/dense/1_cluster/finetune.opt.c4.1.3b.0edr.mu10000.wu0.bsz8.uf1.fp16adam.rs1234.lr2e-05.pat_10000.ngpu4/consolidated.pt'
    logger.info(f"initializing MoE from OPT checkpoint at {path_to_opt}....")
    opt_state = torch.load(path_to_opt)
    opt_state['extra_state']["train_iterator"]['epoch'] = 1
    opt_state['extra_state']["train_iterator"]['iterations_in_epoch'] = 0
    expert_state_dict = torch.load('/gscratch/zlab/sg01/en_moe_lm_15b/model-rank-0.pt')
    stream_ = torch.load('/gscratch/zlab/sg01/en_moe_lm_15b/model-rank-0.pt')
    expert_cfg = expert_state_dict['cfg']
    
    orig_model_cfg = vars(opt_state['cfg']['model'])
    for key in set(expert_cfg['model'].keys()) - set(orig_model_cfg.keys()):
        orig_model_cfg[key] = expert_cfg['model'][key]
    orig_model_cfg['moe_expert_count'] = num_experts
    opt_state['cfg']['model'] = OmegaConf.create(orig_model_cfg)

    # experts = {}
    keys = list(opt_state['model'].keys())
    for key in keys:
        layer = re.findall(r"decoder.layers.(\d+)", key)
        if layer and ("fc1" in key or "fc2" in key) and int(layer[0]) % 2 != 0:
            if "fc1" in key:
                new_key = re.sub(".fc1", '.moe_layer.experts.0.fc1', key)
            elif "fc2" in key:
                new_key = re.sub(".fc2", '.moe_layer.experts.0.fc2', key)
            opt_state['model'][new_key] = opt_state['model'].pop(key)

    num_layers = opt_state['cfg']['model'].decoder_layers
    hidden_dim = opt_state['cfg']['model'].decoder_embed_dim
    for layer in range(num_layers):
        if layer % 2 != 0:
            opt_state['model'][f'decoder.layers.{layer}.moe_layer.gate.wg.weight'] = torch.rand(num_experts, hidden_dim).float().half()
    
    expert_state = opt_state.copy()
    shared_state = opt_state.copy()
    for key in opt_state['model'].keys():
        if "expert" in key:
            expert_state['model'][key] = opt_state['model'][key]
        else:
            shared_state['model'][key] = opt_state['model'][key]
    shared_state['model']['decoder.output_projection.weight'] = opt_state['model']['decoder.embed_tokens.weight']
    if not Path(output_dir).is_dir():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i in range(num_experts):
        torch.save(shared_state, Path(output_dir) / f"checkpoint_last-shared-shard{i}.pt")
        torch.save(expert_state, Path(output_dir) / f"checkpoint_last-rank-{i}-shard{i}.pt")
    
    # hidden_dim = opt_state['cfg']['model'].decoder_embed_dim
    # num_layers = opt_state['cfg']['model'].decoder_layers
    # loop through the OPT state dict
    # for key in opt_state['model'].keys():
    #     # get layer of the key
    #     layer = re.findall(r"decoder.layers.(\d+)", key)
    #     if not layer or int(layer[0]) % 2 == 0 or 'fc' not in key:
    #         # if the key doesnt have a layer, or is an even layer - add to shared state
    #         shared_state[key] = opt_state['model'][key]
    #     else:
    #         new_key = re.sub(".fc1", '.moe_layer.experts.0.fc1', key)
    #         new_key = re.sub(".fc2", '.moe_layer.experts.0.fc2', new_key)
    #         expert_state[new_key] = opt_state['model'][key]
    # for layer in range(num_layers):
    #     if layer % 2 != 0:
    #         shared_state[f'decoder.layers.{layer}.moe_layer.gate.wg.weight'] = torch.rand(num_experts, hidden_dim).float().half()
    # # shared_state['decoder.embed_positions._float_tensor'] = torch.Tensor([1.]).float().half()
    # shared_state['decoder.output_projection.weight'] = opt_state['model']['decoder.embed_tokens.weight']
    
    # from metaseq import pdb; pdb.set_trace()
    # _ = shared_state.pop('decoder.embed_positions.weight')


    # res_shared = opt_state.copy()
    # res_shared['model'] = shared_state
    # res_shared['cfg'] = expert_cfg
    # # from fairseq import pdb; pdb.set_trace()
    # res_shared['cfg']['model']['moe_expert_count'] = num_experts
    
    # # res_shared['cfg']['model']['decoder_learned_pos'] = True
    # # res_shared['cfg']['task']['merges_filename'] = "/gscratch/zlab/sg01/fairseq/gpt2_bpe/vocab.bpe"
    # # res_shared['cfg']['task']['vocab_filename'] = "/gscratch/zlab/sg01/fairseq/gpt2_bpe/encoder.json"

    # res_experts = []
    # for i in range(num_experts):
    #     res_expert = opt_state.copy()
    #     res_expert['cfg'] = expert_cfg
    #     res_expert['model'] = expert_state
    #     res_experts.append(res_expert)
    # if not Path(output_dir).is_dir():
    #     Path(output_dir).mkdir(parents=True, exist_ok=True)
    # for i in range(num_experts):
    #     torch.save(res_shared, Path(output_dir) / f"checkpoint_last-shared-shard{i}.pt")
    #     torch.save(res_experts[i], Path(output_dir) / f"checkpoint_last-rank-{i}-shard{i}.pt")
    return


    # for i in range(num_layers):
    #     if "self_attn" in 
    #     if i % 2 != 0:
    #         new_key = re.sub(f"moe_layer.experts.\d+.", '', key)

    # for key in shared_state['model']:                                                                                                                                            
    #     if opt_state['model'].get(key) is not None:
    #         shared_state['model'][key] = opt_state['model'][key]
    # for key in expert_state['model']:
    #     new_key = re.sub(f"moe_layer.experts.\d+.", '', key)
    #     if opt_state['model'].get(new_key) is not None:
    #         expert_state['model'][key] = opt_state['model'][new_key]


def load_expert_state(local_path):
    checkpoint_files_count = len(glob(re.sub('rank-[0-9]+', 'rank-*', local_path)))
    world_size = distributed_utils.get_data_parallel_world_size()
    rank = distributed_utils.get_data_parallel_rank()
    if world_size < checkpoint_files_count:
        assert checkpoint_files_count % world_size == 0
        logger.info(
            f"Found total {checkpoint_files_count} expert files and"
            f" current distributed world size: {world_size},"
            " Stitching experts to able to load on current world size."
        )
        local_expert_count = int(checkpoint_files_count / world_size)
        start_rank = local_expert_count * rank
        expert_states = []
        for expert_rank in range(start_rank, start_rank + local_expert_count):
            fname = re.sub(
                'rank-[0-9]+',
                'rank-{0}'.format(expert_rank),
                local_path,
            )
            expert_states.append(torch_load_cpu(fname))
        expert_state = merge_multi_local_expert_states(expert_states)
    else:
        expert_state = torch_load_cpu(local_path)
    return expert_state


def assert_equal(a, b, msg=''):
    assert a == b, f"{msg}{a} != {b}"


def _unflat_expert_tensor_state(expert, shared) -> Dict:
    """called from merge_expert_and_shared_state, for FSDP only."""

    local_to_globals = defaultdict(list)
    for global_id, local_id in shared['param_id_map'].items():
        if local_id in shared['uncollected_local_ids']:
            local_to_globals[local_id].append(global_id)

    flat_expert_state = expert['state']
    unflat_state = {}
    for local_id, global_ids in local_to_globals.items():
        global_ids = sorted(global_ids)
        unflat_state.update({g: {} for g in global_ids})
        already_unflat = {k: v for k, v in flat_expert_state[local_id].items() if not torch.is_tensor(v) or is_singleton_tensor(v)}
        for buffer_name, flat_param in flat_expert_state[local_id].items():
            if torch.is_tensor(flat_param) and not is_singleton_tensor(flat_param):
                unflat_shapes = [shared['state'][g][buffer_name].shape for g in global_ids]
                numels = [np.prod(s) for s in unflat_shapes]
                unflat = zip(global_ids, (t.view(s) for (t, s) in zip(flat_param.split(numels), unflat_shapes)))
                for gid, t in unflat:
                    unflat_state[gid][buffer_name] = t
                    unflat_state[gid].update(already_unflat)
    return unflat_state
