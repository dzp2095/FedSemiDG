import logging
import os
import subprocess

import torch

from src.utils.args_parser import args


def get_visible_list():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is None or visible.strip() == "":
        count = torch.cuda.device_count()
        return list(range(count)), {i: i for i in range(count)}

    phys_list = [int(x.strip()) for x in visible.split(",") if x.strip()]
    mapping = {physical: logical for logical, physical in enumerate(phys_list)}
    return phys_list, mapping


def _query_free_memory_mb():
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "nvidia-smi failed")
    return [int(line.strip()) for line in proc.stdout.splitlines() if line.strip()]


def get_free_gpu(gpu_exclude_list=None):
    if gpu_exclude_list is None:
        gpu_exclude_list = []

    physical_visible, physical_to_logical = get_visible_list()
    memory_free = _query_free_memory_mb()

    candidates = [
        (physical_idx, memory_free[physical_idx])
        for physical_idx in physical_visible
        if physical_idx not in gpu_exclude_list and physical_idx < len(memory_free)
    ]
    if not candidates:
        raise RuntimeError("No available GPU found after applying visibility and exclusion filters.")

    best_physical, _ = max(candidates, key=lambda item: item[1])
    return physical_to_logical[best_physical]


def get_free_device_name():
    if not (torch.cuda.is_available() and args.gpu):
        return "cpu"

    try:
        gpu_logic = get_free_gpu(gpu_exclude_list=args.gpu_exclude_list)
        logging.info("Using GPU (logical id): %s", gpu_logic)
        return f"cuda:{gpu_logic}"
    except Exception as exc:
        logging.warning("Falling back to CPU because GPU selection failed: %s", exc)
        return "cpu"
