import yaml
import logging

import random
import numpy as np
import torch
from torch.backends import cudnn
from pathlib import Path
from datetime import datetime

from src.modules.supervised_trainer import SupervisedTrainer
from src.modules.semi_trainer import SemiTrainer
from src.utils.args_parser import args, args2cfg

if args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

now = str(datetime.timestamp(datetime.now()))

log_dir = Path(f'/storage/zhipengdeng/data/segmentation/fed_semi/log')
Path(log_dir).mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=f'{log_dir}/log_{args.run_name}_{now}.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

if __name__ == "__main__":

    try:
        cfg = yaml.safe_load(open(args.config))
        cfg["train"]["checkpoint_dir"] = f'{cfg["train"]["checkpoint_dir"]}{now}'
        cfg = args2cfg(cfg, args)
        train_path = cfg['dataset']['train']
        logging.info(f"Training data path: {train_path}")
        total_rounds = cfg['fl']['rounds']
        iter_per_round = cfg['fl']['local_iter']
        max_iter =  total_rounds * iter_per_round
        cfg["train"]["max_iter"] = max_iter
        if args.trainer == 'semi':
            trainer = SemiTrainer(args, cfg)
        else:
            trainer = SupervisedTrainer(args, cfg)
        trainer.train(max_iter)

    except Exception as e:
        logging.critical(e, exc_info=True)
