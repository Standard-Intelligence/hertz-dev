import os
import torch as T
import re
from tqdm import tqdm
from datetime import timedelta

import requests
import hashlib

from io import BytesIO  
from huggingface_hub import hf_hub_download

def rank0():
    rank = os.environ.get('RANK')
    if rank is None or rank == '0':
        return True
    else:
        return False
    
def local0():
    local_rank = os.environ.get('LOCAL_RANK')
    if local_rank is None or local_rank == '0':
        return True
    else:
        return False
class tqdm0(tqdm):
    def __init__(self, *args, **kwargs):
        total = kwargs.get('total', None)
        if total is None and len(args) > 0:
            try:
                total = len(args[0])
            except TypeError:
                pass
        if total is not None:
            kwargs['miniters'] = max(1, total // 20)
        super().__init__(*args, **kwargs, disable=not rank0(), bar_format='{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]')
    
def print0(*args, **kwargs):
    if rank0():
        print(*args, **kwargs)

_PRINTED_IDS = set()

def printonce(*args, id=None, **kwargs):
    if id is None:
        id = ' '.join(map(str, args))
    
    if id not in _PRINTED_IDS:
        print(*args, **kwargs)
        _PRINTED_IDS.add(id)

def print0once(*args, **kwargs):
    if rank0(): 
        printonce(*args, **kwargs)

def init_dist():
    if T.distributed.is_initialized():
        print0('Distributed already initialized')
        rank = T.distributed.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = T.distributed.get_world_size()
    else:
        try:
            rank = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            device = f'cuda:{local_rank}'
            T.cuda.set_device(device)
            T.distributed.init_process_group(backend='nccl', timeout=timedelta(minutes=30), rank=rank, world_size=world_size, device_id=T.device(device))
            print(f'Rank {rank} of {world_size}.')
        except Exception as e:
            print0once(f'Not initializing distributed env: {e}')
            rank = 0
            local_rank = 0
            world_size = 1
    return rank, local_rank, world_size

def load_ckpt(load_from_location, expected_hash=None):
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1' #Disable this to speed up debugging errors with downloading from the hub
    if local0():
        repo_id = "si-pbc/hertz-dev"
        print0(f'Loading checkpoint from repo_id {repo_id} and filename {load_from_location}.pt. This may take a while...')
        save_path = hf_hub_download(repo_id=repo_id, filename=f"{load_from_location}.pt")
        print0(f'Downloaded checkpoint to {save_path}')
        if expected_hash is not None:
            with open(save_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash != expected_hash:
                print(f'Hash mismatch for {save_path}. Expected {expected_hash} but got {file_hash}. Deleting checkpoint and trying again.')
                os.remove(save_path)
                return load_ckpt(load_from_location, expected_hash)
    if T.distributed.is_initialized():
        save_path = [save_path]
        T.distributed.broadcast_object_list(save_path, src=0)
        save_path = save_path[0]
    loaded = T.load(save_path, weights_only=False, map_location='cpu')
    print0(f'Loaded checkpoint from {save_path}')
    return loaded