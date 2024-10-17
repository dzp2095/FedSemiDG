import yaml
import logging

import random
import numpy as np
import torch
from torch.backends import cudnn
from pathlib import Path
from datetime import datetime

from src.fl.client import Client
from src.fl.server import Server
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
        if args.eval_only:
            server = Server([], cfg)
            cfg["train"]["resume_path"] = args.resume_path
            server.global_test()
        else:
            if args.trainer == 'semi':
                clients = []
                labeled_clients = args.labeled_clients
                unseen_client = args.unseen_client
                duplicate_clients = set.intersection(set(labeled_clients), set([unseen_client]))
                if duplicate_clients:
                    raise ValueError(f"Duplicate clients: {duplicate_clients}")
                logging.info(f"Labeled clients: {labeled_clients}, Unseen clients: {unseen_client}")
                for client in labeled_clients:
                    clients.append(Client(client, args, cfg, is_labeled_client=True))
            else:
                clients = []
                labeled_clients = args.labeled_clients
                unseen_client = args.unseen_client
                logging.info(f"Labeled clients: {labeled_clients}, Unseen clients: {unseen_client}")
                for client in labeled_clients:
                    clients.append(Client(client, args, cfg, is_labeled_client=True, is_fully_supervised=True))                                                                             
            server = Server(clients, unseen_client, cfg)
            server.run()
    except Exception as e:
        logging.critical(e, exc_info=True)
