import abc
import copy
import logging
import os

from src.modules.semi_trainer import SemiTrainer
from src.modules.supervised_trainer import SupervisedTrainer


class Client(abc.ABC):
    def __init__(self, name, args, cfg, is_labeled_client=True, is_fully_supervised=False):
        if not is_labeled_client:
            raise ValueError("Unlabeled-only FL clients are no longer supported.")

        self._name = name
        self.cfg = copy.deepcopy(cfg)
        self.setup_before_trainer()

        if is_fully_supervised:
            self.trainer = SupervisedTrainer(args, self.cfg)
        else:
            self.trainer = SemiTrainer(args, self.cfg)

        self.setup_after_trainer()
        self.round = 0
        self._is_labeled_client = is_labeled_client

    def load_model(self, model_weights):
        self.trainer.load_model(model_weights)

    def setup_before_trainer(self):
        run_name = self.cfg["wandb"].get("run_name", "fedsemi")
        rounds = self.cfg["fl"].get("rounds", 1)
        self.cfg["wandb"]["run_name"] = f"{run_name}_rounds_{rounds}_{self.name}"

        self.cfg["dataset"]["train"] = os.path.join(self.cfg["dataset"]["train"], self.name)
        self._train_path = self.cfg["dataset"]["train"]
        logging.info("%s: training path = %s", self.name, self._train_path)

        self.total_rounds = int(self.cfg["fl"].get("rounds", 1))

    def setup_after_trainer(self):
        iter_cfg = self.cfg["local"]["iter_per_round"]
        if iter_cfg.get("epoch") is not None:
            epoch = int(iter_cfg["epoch"])
            self.iter_per_round = ((self.train_data_num // self.cfg["train"]["batch_size"]) + 1) * epoch
        else:
            self.iter_per_round = int(iter_cfg["iter"])

        self.trainer.max_iter = self.total_rounds * self.iter_per_round

    def run(self):
        logging.info("%s: start round %d", self.name, self.round)
        self.trainer.train(self.iter_per_round)
        self.round += 1

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
    def train_path(self):
        return self._train_path

    @property
    def is_labeled_client(self):
        return self._is_labeled_client
