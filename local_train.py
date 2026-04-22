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

from src.modules.semi_trainer import SemiTrainer
from src.modules.supervised_trainer import SupervisedTrainer
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

        train_path = cfg.get("dataset", {}).get("train")
        logging.info("Training data path: %s", train_path)

        total_rounds = cfg.get("fl", {}).get("rounds", 1)
        iter_per_round = cfg.get("fl", {}).get("local_iter", 1)
        max_iter = total_rounds * iter_per_round

        if args.max_iter_override is not None:
            max_iter = args.max_iter_override

        cfg["train"]["max_iter"] = max_iter

        if args.trainer == "semi":
            trainer = SemiTrainer(args, cfg)
        else:
            trainer = SupervisedTrainer(args, cfg)

        trainer.train(max_iter)
    except Exception as exc:
        logging.critical(exc, exc_info=True)
        raise
