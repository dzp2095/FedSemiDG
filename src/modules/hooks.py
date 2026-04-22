import copy
import logging
import os
from datetime import datetime

import torch
from torch.utils.data import ConcatDataset

from src.model.ema import ModelEMA
from src.modules.defaults import HookBase
from src.tasks.task_registry import TaskRegistry

try:
    import wandb
except Exception:
    wandb = None


class Timer(HookBase):
    def before_train(self):
        self.tick = datetime.now()
        logging.info("Training started at %s", self.tick.strftime("%Y-%m-%d %H:%M:%S"))

    def after_train(self):
        tock = datetime.now()
        logging.info("Training finished at %s", tock.strftime("%Y-%m-%d %H:%M:%S"))
        logging.info("Elapsed: %s", str(tock - self.tick).split(".")[0])


class WAndBUploader(HookBase):
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.enabled = bool(cfg.get("hooks", {}).get("wandb", False))
        self.experiment = None

    def before_train(self):
        if not self.enabled:
            return
        if wandb is None:
            logging.warning("wandb is not installed; disable wandb hook.")
            self.enabled = False
            return

        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)

        mode = os.environ.get("WANDB_MODE") or ("online" if api_key else "disabled")
        self.experiment = wandb.init(
            project=self.cfg.get("wandb", {}).get("project", "fedsemi"),
            name=self.cfg.get("wandb", {}).get("run_name", "fedsemi"),
            resume="allow",
            mode=mode,
        )
        self.log_interval = max(1, int(self.cfg.get("train", {}).get("log_interval", 50)))
        self.experiment.config.update(
            dict(
                steps=int(self.trainer.max_iter),
                batch_size=int(self.cfg["train"]["batch_size"]),
                learning_rate=float(self.cfg["train"]["optimizer"]["lr"]),
            ),
            allow_val_change=True,
        )

    def after_step(self):
        if self.experiment is None:
            return
        if self.trainer.iter % self.log_interval != 0:
            return

        metrics = dict(self.trainer.metric_logger._dict)
        if metrics:
            self.experiment.log(metrics)

    def after_train(self):
        if self.experiment is not None:
            self.experiment.finish()


class GA(HookBase):
    def __init__(self, cfg):
        self._ga_value = 1.0
        self.cfg = copy.deepcopy(cfg)
        self.factory = TaskRegistry.get_factory(cfg["task"])
        self.evaluation_strategy = self.factory.create_evaluation_strategy(cfg)

    def before_train(self):
        self.global_model = copy.deepcopy(self.trainer.model)
        return super().before_train()

    def after_train(self):
        train_root = self.cfg["dataset"]["train"]
        cfg = copy.deepcopy(self.cfg)

        cfg["dataset"]["only_image"] = os.path.join(train_root, "labeled.csv")
        labeled_dataset = self.factory.create_dataset(mode="only_image", cfg=cfg)

        cfg["dataset"]["only_image"] = os.path.join(train_root, "unlabeled.csv")
        unlabeled_dataset = self.factory.create_dataset(mode="only_image", cfg=cfg)

        dataset = ConcatDataset([labeled_dataset, unlabeled_dataset])
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(self.cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=int(self.cfg["train"]["num_workers"]),
            pin_memory=True,
        )

        self.ga_value = self.evaluation_strategy.cal_kl_loss(
            self.trainer.model,
            self.global_model,
            data_loader,
            self.trainer.device,
        )
        return super().after_train()

    @property
    def ga_value(self):
        return self._ga_value

    @ga_value.setter
    def ga_value(self, value):
        self._ga_value = float(value)


class EMA(HookBase):
    def __init__(self, cfg):
        self.decay = float(cfg["train"]["ema_decay"])

    def before_train(self):
        self.trainer.dynamic_teacher = ModelEMA(self.trainer.device, self.trainer.model, self.decay)

    def after_step(self):
        self.trainer.dynamic_teacher.update(self.trainer.model)
