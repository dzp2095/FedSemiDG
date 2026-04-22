import logging
import os
import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml
from torch.backends import cudnn

from src.fl.client import Client
from src.fl.server import Server
from src.utils.args_parser import args, args2cfg


if args.gpu_pool is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_pool


def setup_seed() -> None:
    if args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)


def setup_logger() -> datetime:
    task_name = os.path.basename(os.path.dirname(args.config))
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    date_time = now.strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name.lower()

    log_dir = Path(f"./log/{task_name}/{run_name}")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_dir / f"{date_time}.log",
        filemode="a",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )
    return now


def build_clients(cfg):
    labeled_clients = args.labeled_clients or []
    if not labeled_clients:
        raise ValueError("--labeled_clients is required for fl_train.py")

    unseen_client = args.unseen_client
    if unseen_client is None:
        raise ValueError("--unseen_client is required for fl_train.py")

    duplicate_clients = set.intersection(set(labeled_clients), {unseen_client})
    if duplicate_clients:
        raise ValueError(f"Duplicate clients in labeled/unseen: {duplicate_clients}")

    logging.info("Labeled clients: %s, Unseen client: %s", labeled_clients, unseen_client)

    if args.trainer == "semi":
        clients = [Client(client, args, cfg, is_labeled_client=True) for client in labeled_clients]
    else:
        clients = [
            Client(client, args, cfg, is_labeled_client=True, is_fully_supervised=True)
            for client in labeled_clients
        ]
    return clients, unseen_client


if __name__ == "__main__":
    setup_seed()
    now = setup_logger()

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cfg = args2cfg(cfg, args)
        cfg.setdefault("train", {})
        checkpoint_dir = cfg["train"].get("checkpoint_dir", "./checkpoints/")
        cfg["train"]["checkpoint_dir"] = f"{checkpoint_dir}{now.strftime('%Y%m%d_%H%M%S')}"

        clients, unseen_client = build_clients(cfg)
        server = Server(clients, unseen_client, cfg)
        server.run()
    except Exception as exc:
        logging.critical(exc, exc_info=True)
        raise
