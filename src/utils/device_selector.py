
import logging
import tempfile

import torch
import os
import numpy as np
from src.utils.args_parser import args

def get_visible_list():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if vis is None or vis.strip() == "":
        # 没设，说明所有卡都可见
        return list(range(torch.cuda.device_count())), {i: i for i in range(torch.cuda.device_count())}
    # e.g. "6,7"  →  [6,7]
    phys_list = [int(x) for x in vis.split(',')]
    # 物理 → 逻辑 的映射表，如 {6:0, 7:1}
    mapping = {p: i for i, p in enumerate(phys_list)}
    return phys_list, mapping

def get_free_gpu(gpu_exclude_list=None):
    if gpu_exclude_list is None:
        gpu_exclude_list = []
    phys_visible, phys2logic = get_visible_list()

    # 用 nvidia-smi 拿到所有物理卡的剩余显存
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_filename = tmp_file.name
    os.system(f'nvidia-smi -q -d Memory |grep -A5 GPU |grep Free > {tmp_filename}')

    with open(tmp_filename, 'r') as f:
        mem_free = [int(x.split()[2]) for x in f.readlines()]
    os.remove(tmp_filename)

    # 只在 **可见** 并且 **不在 exclude** 的卡里选
    candidates = [
        (idx, mem_free[idx])
        for idx in phys_visible
        if idx not in gpu_exclude_list
    ]
    best_phys, _ = max(candidates, key=lambda x: x[1])
    best_logic = phys2logic[best_phys]        # ← 关键映射
    return best_logic                         # 返回逻辑 ID (0..len(visible)-1)

def get_free_device_name():
    if torch.cuda.is_available() and args.gpu:
        gpu_logic = get_free_gpu(gpu_exclude_list=args.gpu_exclude_list)
        logging.info(f'Using GPU (logic id): {gpu_logic}')
        return f'cuda:{gpu_logic}'
    else:
        return 'cpu'
