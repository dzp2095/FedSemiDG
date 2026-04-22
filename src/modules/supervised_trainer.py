import copy
import os

import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss

from src.datasets.sampler import TrainingSampler
from src.model.mit_PLD_b4 import mit_PLD_b4
from src.model.unet import UNet
from src.modules import hooks
from src.modules.defaults import TrainerBase
from src.tasks.task_registry import TaskRegistry


class SupervisedTrainer(TrainerBase):
    """Supervised local trainer for the public four-dataset setting."""

    def __init__(self, args, cfg) -> None:
        super().__init__(args, cfg)
        self.register_hooks(self.build_hooks())

        task = cfg["task"]
        if task == "cardiac":
            self.run_step = self._run_step_cardiac
        elif task == "spine":
            self.run_step = self._run_step_spine
        elif task == "bladder":
            self.run_step = self._run_step_bladder
        elif task == "colon":
            self.run_step = self._run_step_colon
        else:
            raise NotImplementedError(f"Task {task} is not supported")

    def build_model(self):
        task = self.cfg["task"]
        num_classes = int(self.cfg["model"]["num_classes"])
        if task == "colon":
            self.model = mit_PLD_b4(class_num=num_classes)
        else:
            num_channels = int(self.cfg["model"]["num_channels"])
            self.model = UNet(num_channels, num_classes)

    def init_dataloader(self):
        batch_size = int(self.cfg["train"]["batch_size"])
        factory = TaskRegistry.get_factory(self.cfg["task"])

        train_root = self.cfg["dataset"]["train"]
        train_csv = os.path.join(train_root, "labeled.csv") if os.path.isdir(train_root) else train_root

        cfg_for_data = copy.deepcopy(self.cfg)
        cfg_for_data["dataset"]["train"] = train_csv
        dataset = factory.create_dataset(mode="train", is_labeled=True, cfg=cfg_for_data)

        self._train_data_num = len(dataset)
        num_workers = int(self.cfg["train"]["num_workers"])
        seed = int(self.cfg["dataset"].get("seed", 42))

        self._data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            sampler=TrainingSampler(len(dataset), seed=seed),
        )
        self._data_iter = iter(self._data_loader)

    def _next_batch(self):
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self._data_loader)
            return next(self._data_iter)

    def load_model(self, model_weights):
        self.model.load_state_dict(model_weights, strict=False)

        old_scheduler = copy.deepcopy(self.lr_scheduler.state_dict())
        old_optimizer = copy.deepcopy(self.optimizer.state_dict())
        self.build_optimizer()
        self.optimizer.load_state_dict(old_optimizer)
        self.build_schedular(self.optimizer)
        self.lr_scheduler.load_state_dict(old_scheduler)

    def build_hooks(self):
        ret = [hooks.Timer()]
        if self.cfg.get("hooks", {}).get("wandb", False):
            ret.append(hooks.WAndBUploader(self.cfg))
        return ret

    def run_step(self):
        raise NotImplementedError

    def _run_step_cardiac(self):
        self.model.train()
        _, image, mask = self._next_batch()

        image = image.to(self.device)
        mask = mask.to(self.device).long()

        output = self.model(image)
        mask = F.one_hot(mask, num_classes=output.shape[1]).permute(0, 3, 1, 2).float().to(self.device)

        loss = DiceCELoss(softmax=True)(output, mask)
        self._backward_step(loss)

    def _run_step_spine(self):
        self.model.train()
        _, image, mask = self._next_batch()

        image = image.to(self.device)
        mask = mask.permute(0, 3, 1, 2).float().to(self.device)

        output = self.model(image)
        loss = DiceCELoss(sigmoid=True)(output, mask)
        self._backward_step(loss)

    def _run_step_bladder(self):
        self.model.train()
        _, image, mask = self._next_batch()

        image = image.to(self.device)
        mask = mask.unsqueeze(1).float().to(self.device)

        output = self.model(image)
        loss = DiceCELoss(sigmoid=True)(output, mask)
        self._backward_step(loss)

    def _run_step_colon(self):
        self.model.train()
        _, image, mask = self._next_batch()

        image = image.to(self.device)
        mask = (mask > 0).float().unsqueeze(1).to(self.device)

        output = self.model(image)
        loss = DiceCELoss(sigmoid=True)(output, mask)
        self._backward_step(loss)

    def _backward_step(self, loss):
        self.loss_logger.update(loss=loss)
        self.metric_logger.update(loss=loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @property
    def train_data_num(self):
        return self._train_data_num
