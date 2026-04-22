import copy
import logging
import os
import weakref
from pathlib import Path
from shutil import copyfile
from typing import List, Optional

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.utils.device_selector import get_free_device_name
from src.utils.metric_logger import EMAMetricLogger, MetricLogger


class HookBase:
    """Base class for trainer hooks."""

    trainer: "TrainerBase" = None

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def state_dict(self):
        return {}


class TrainerBase:
    """Base class for iterative training with hook support."""

    def __init__(self, args, cfg) -> None:
        self._hooks: List[HookBase] = []
        self.args = args
        self.cfg = copy.deepcopy(cfg)

        self.name = "defaults"
        self.metric_logger = MetricLogger()
        self.loss_logger = EMAMetricLogger()

        self.init_dataloader()
        self.setup_train()

    def setup_train(self):
        self.iter = 0
        self.start_iter = 0
        self.max_iter = int(self.cfg["train"]["max_iter"])
        self.device = get_free_device_name()

        self.build_model()
        self.model = self.model.to(self.device)

        checkpoint_dir = Path(self.cfg["train"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.args.config and os.path.isfile(self.args.config):
            cfg_file = checkpoint_dir / os.path.basename(self.args.config)
            copyfile(self.args.config, cfg_file)

        self.build_optimizer()
        self.build_schedular(self.optimizer)

    def build_model(self):
        raise NotImplementedError

    def build_optimizer(self):
        optimizer_cfg = self.cfg["train"]["optimizer"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=optimizer_cfg["lr"],
            betas=(optimizer_cfg["beta1"], optimizer_cfg["beta2"]),
            weight_decay=optimizer_cfg["weight_decay"],
        )

    def build_schedular(self, optimizer):
        scheduler_cfg = self.cfg["train"]["lr_scheduler"]
        self.lr_scheduler = ReduceLROnPlateau(
            optimizer,
            factor=scheduler_cfg["factor"],
            patience=scheduler_cfg["patience"],
            min_lr=scheduler_cfg["min_lr"],
        )

    def init_dataloader(self):
        raise NotImplementedError

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        hooks = [h for h in hooks if h is not None]
        for hook in hooks:
            assert isinstance(hook, HookBase)
            hook.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, train_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration %s", self.start_iter)
        start_iter = self.start_iter
        max_iter = self.max_iter

        with tqdm(total=train_iter) as pbar:
            try:
                self.before_train()
                for self.iter in range(start_iter, min(max_iter, start_iter + train_iter)):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    pbar.update(1)

                    if "loss" in self.metric_logger._dict:
                        pbar.set_postfix({"loss": f"{self.metric_logger._dict['loss']:.4f}"})

                self.iter += 1
            except Exception:
                logger.exception("Exception during training")
                raise
            finally:
                self.start_iter = self.iter
                self.after_train()

    def before_train(self):
        for hook in self._hooks:
            hook.before_train()

    def after_train(self):
        for hook in self._hooks:
            hook.after_train()

    def before_step(self):
        for hook in self._hooks:
            hook.before_step()

    def after_step(self):
        for hook in self._hooks:
            hook.after_step()

    def run_step(self):
        raise NotImplementedError
