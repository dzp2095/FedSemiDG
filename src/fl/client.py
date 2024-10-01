import copy
import torch
import abc
import logging
from src.tasks.task_registry import TaskRegistry
from src.utils.device_selector import get_free_device_name

from src.modules.semi_trainer import SemiTrainer
from src.modules.unsupervised_trainer import UnsupervisedTrainer
from src.modules.supervised_trainer import SupervisedTrainer

class Client(abc.ABC):
    def __init__(self, name, args, cfg, is_labeled_client=True, is_fully_supervised=False):
        self._name = name
        self.cfg = copy.deepcopy(cfg)
        self.setup()
        self.global_discriminator = None
        if is_labeled_client:
            if is_fully_supervised:
                self.trainer = SupervisedTrainer(args, self.cfg)
            else:
                self.trainer = SemiTrainer(args, self.cfg)
        else:
            self.trainer = UnsupervisedTrainer(args, self.cfg)
        self.round = 0
        self.trainer.client_label = int(name.split('_')[-1]) + 1
        self._is_labeled_client = is_labeled_client

    def load_model(self, model_weights):
        self.trainer.load_model(model_weights)

    def setup(self):
        self.cfg["wandb"]["run_name"] = f"{self.cfg['wandb']['run_name']}_rounds_{self.cfg['fl']['rounds']}_{self.name}"
        self.cfg['dataset']['train'] = f"{self.cfg['dataset']['train']}/{self.name}"
        self._train_path = self.cfg['dataset']['train']
        logging.info(f"{self.name}: Training data path: {self._train_path}")
        self.total_rounds = self.cfg['fl']['rounds']
        if self.cfg['local']['iter_per_round']['epoch'] != None:
            epoch = self.cfg['local']['iter_per_round']['epoch']
            self.iter_per_round = (self.train_data_num // self.cfg['train']['batch_size']) * epoch
        else:
            self.iter_per_round = self.cfg['local']['iter_per_round']['iter']
        max_iter =  self.total_rounds * self.iter_per_round
        self.cfg["train"]["max_iter"] = max_iter
    
    def run(self):
        """train the model """
        logging.info(f"{self.name}: Start training from round {self.round}")
        self.trainer.train(self.iter_per_round)
        self.round+=1

    @property
    def model(self):
        return self.trainer.model

    @property
    def name(self):
        return self._name

    @property
    def train_data_num(self):
        return self.trainer.train_data_num
        
    @property
    def class_nums(self):
        return self.trainer._class_nums
    
    @property
    def train_path(self):
        return self._train_path

    @property
    def is_labeled_client(self):
        return self._is_labeled_client

