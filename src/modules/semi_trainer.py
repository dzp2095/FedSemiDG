import copy
import os

import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss, MaskedDiceLoss

from src.datasets.sampler import TrainingSampler
from src.model.mit_PLD_b4 import mit_PLD_b4
from src.model.unet import UNet
from src.modules import hooks
from src.modules.defaults import TrainerBase
from src.tasks.task_registry import TaskRegistry


class SemiTrainer(TrainerBase):
    """Semi-supervised trainer with FGASL mainline logic (GAA+DR+PIA)."""

    def __init__(self, args, cfg, is_fully_supervised=True) -> None:
        super().__init__(args, cfg)
        self.register_hooks(self.build_hooks())
        self._is_fully_supervised = is_fully_supervised

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
            raise NotImplementedError(f"Task {task} not supported")

        self.dynamic_teacher = None
        self.fixed_teacher = None
        self.uncertain_threshold = 0.0

    def build_model(self):
        task = self.cfg["task"]
        num_classes = int(self.cfg["model"]["num_classes"])
        num_channels = int(self.cfg["model"]["num_channels"])
        fp_rate = float(self.cfg["model"]["fp_rate"])

        if task == "colon":
            self.model = mit_PLD_b4(class_num=num_classes, fp_rate=fp_rate)
        else:
            self.model = UNet(num_channels, num_classes, fp_rate=fp_rate)

    def init_dataloader(self):
        factory = TaskRegistry.get_factory(self.cfg["task"])
        train_root = self.cfg["dataset"]["train"]

        self.cfg["dataset"]["train"] = os.path.join(train_root, "labeled.csv")
        labeled_dataset = factory.create_dataset(mode="train", is_labeled=True, cfg=self.cfg)

        self.cfg["dataset"]["train"] = os.path.join(train_root, "unlabeled.csv")
        unlabeled_dataset = factory.create_dataset(mode="train", is_labeled=False, cfg=self.cfg)

        self.cfg["dataset"]["train"] = train_root
        self.labeled_data_num = len(labeled_dataset)
        self.unlabeled_data_num = len(unlabeled_dataset)
        self._train_data_num = self.labeled_data_num + self.unlabeled_data_num

        batch_size = max(1, int(self.cfg["train"]["batch_size"]) // 2)
        num_workers = int(self.cfg["train"]["num_workers"])
        seed = int(self.cfg["dataset"]["seed"])

        self._labeled_data_loader = torch.utils.data.DataLoader(
            labeled_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            sampler=TrainingSampler(len(labeled_dataset), seed=seed),
        )
        self._unlabeled_data_loader = torch.utils.data.DataLoader(
            unlabeled_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            sampler=TrainingSampler(len(unlabeled_dataset), seed=seed),
        )

        self._labeled_data_iter = iter(self._labeled_data_loader)
        self._unlabeled_data_iter = iter(self._unlabeled_data_loader)

    def _next_labeled_batch(self):
        try:
            return next(self._labeled_data_iter)
        except StopIteration:
            self._labeled_data_iter = iter(self._labeled_data_loader)
            return next(self._labeled_data_iter)

    def _next_unlabeled_batch(self):
        try:
            return next(self._unlabeled_data_iter)
        except StopIteration:
            self._unlabeled_data_iter = iter(self._unlabeled_data_loader)
            return next(self._unlabeled_data_iter)

    def _next_train_batches(self):
        _, lb_x, lb_y = self._next_labeled_batch()
        _, ulb_w, ulb_s = self._next_unlabeled_batch()
        return lb_x, lb_y, ulb_w, ulb_s

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

        ret.append(hooks.EMA(self.cfg))
        ret.append(hooks.GA(self.cfg))
        return ret

    def before_train(self):
        self.fixed_teacher = copy.deepcopy(self.model)
        for param in self.fixed_teacher.parameters():
            param.requires_grad_(False)
        self.fixed_teacher.eval()
        return super().before_train()

    def after_train(self):
        return super().after_train()

    def run_step(self):
        raise NotImplementedError

    def _entropy_ratio(self):
        start = float(self.cfg["train"]["entropy_start_ratio"])
        end = float(self.cfg["train"]["entropy_end_ratio"])
        if self.max_iter <= 0:
            return end
        return start + (end - start) * float(self.iter) / float(self.max_iter)

    def _update_uncertain_threshold(self, lb_entropy_threshold):
        decay = float(self.cfg["train"]["uncertain_ema_decay"])
        self.uncertain_threshold = decay * self.uncertain_threshold + (1 - decay) * lb_entropy_threshold

    def _forward_with_feature_loss(self, lb_x, ulb_s):
        data = torch.cat((lb_x, ulb_s), dim=0)
        lb_size = lb_x.size(0)

        output, feature, feature_p = self.model(data, return_features=True)
        feature_loss = F.mse_loss(feature, feature_p)

        output_lb_x, output_ulb_s = output[:lb_size], output[lb_size:]
        return output_lb_x, output_ulb_s, feature_loss

    def _build_multiclass_pseudo(self, output_lb_x, ulb_w):
        epsilon = 1e-8

        with torch.no_grad():
            lb_probs = torch.softmax(output_lb_x, dim=1)
            lb_entropy = -(lb_probs * torch.log(lb_probs + epsilon)).sum(dim=1)
            lb_entropy_threshold = torch.quantile(lb_entropy.view(-1), self._entropy_ratio())
            self._update_uncertain_threshold(lb_entropy_threshold)

            ulb_dynamic_logits = self.dynamic_teacher.ema(ulb_w)
            ulb_fixed_logits = self.fixed_teacher(ulb_w)

            ulb_dynamic_probs = torch.softmax(ulb_dynamic_logits, dim=1)
            ulb_fixed_probs = torch.softmax(ulb_fixed_logits, dim=1)
            ulb_probs = (ulb_dynamic_probs + ulb_fixed_probs) / 2
            pseudo_class = torch.argmax(ulb_probs, dim=1)

            entropy_dynamic = -(ulb_dynamic_probs * torch.log(ulb_dynamic_probs + epsilon)).sum(dim=1)
            entropy_fixed = -(ulb_fixed_probs * torch.log(ulb_fixed_probs + epsilon)).sum(dim=1)
            pseudo_mask = ((entropy_dynamic + entropy_fixed) / 2).le(self.uncertain_threshold).float()

            pseudo_one_hot = F.one_hot(pseudo_class, num_classes=ulb_dynamic_logits.shape[1]).permute(0, 3, 1, 2).float()

        return pseudo_class, pseudo_one_hot, pseudo_mask

    def _build_binary_pseudo(self, output_lb_x, ulb_w):
        epsilon = 1e-8

        with torch.no_grad():
            lb_probs = torch.sigmoid(output_lb_x).clamp(min=epsilon, max=1 - epsilon)
            lb_entropy = -(lb_probs * torch.log(lb_probs + epsilon) + (1 - lb_probs) * torch.log(1 - lb_probs + epsilon))
            lb_entropy_threshold = torch.quantile(lb_entropy.view(-1), self._entropy_ratio())
            self._update_uncertain_threshold(lb_entropy_threshold)

            ulb_dynamic_logits = self.dynamic_teacher.ema(ulb_w)
            ulb_fixed_logits = self.fixed_teacher(ulb_w)

            ulb_dynamic_probs = torch.sigmoid(ulb_dynamic_logits).clamp(min=epsilon, max=1 - epsilon)
            ulb_fixed_probs = torch.sigmoid(ulb_fixed_logits).clamp(min=epsilon, max=1 - epsilon)
            ulb_probs = (ulb_dynamic_probs + ulb_fixed_probs) / 2
            pseudo_label = ulb_probs.ge(0.5).float()

            entropy_dynamic = -(ulb_dynamic_probs * torch.log(ulb_dynamic_probs + epsilon) + (1 - ulb_dynamic_probs) * torch.log(1 - ulb_dynamic_probs + epsilon))
            entropy_fixed = -(ulb_fixed_probs * torch.log(ulb_fixed_probs + epsilon) + (1 - ulb_fixed_probs) * torch.log(1 - ulb_fixed_probs + epsilon))
            pseudo_mask = ((entropy_dynamic + entropy_fixed) / 2).le(self.uncertain_threshold).float()

        return pseudo_label, pseudo_mask

    def _compute_binary_unsup_loss(self, output_ulb_s, pseudo_label, pseudo_mask):
        ce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        masked_dice_loss_fn = MaskedDiceLoss(sigmoid=True)

        unsup_loss = (ce_loss_fn(output_ulb_s, pseudo_label) * pseudo_mask).mean()
        for idx in range(output_ulb_s.size(1)):
            unsup_loss += masked_dice_loss_fn(
                output_ulb_s[:, idx, :, :].unsqueeze(1),
                pseudo_label[:, idx, :, :].unsqueeze(1),
                pseudo_mask[:, idx, :, :].unsqueeze(1),
            )
        return unsup_loss

    def _optimize(self, loss):
        self.loss_logger.update(loss=loss)
        self.metric_logger.update(loss=loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _run_step_binary(self, label_transform):
        feature_loss_weight = float(self.cfg["train"]["feature_loss_weight"])

        lb_x, lb_y, ulb_w, ulb_s = self._next_train_batches()
        lb_x = lb_x.to(self.device)
        lb_y = label_transform(lb_y).to(self.device)
        ulb_w = ulb_w.to(self.device)
        ulb_s = ulb_s.to(self.device)

        output_lb_x, output_ulb_s, feature_loss = self._forward_with_feature_loss(lb_x, ulb_s)

        sup_loss = DiceCELoss(sigmoid=True)(output_lb_x, lb_y)
        pseudo_label, pseudo_mask = self._build_binary_pseudo(output_lb_x, ulb_w)
        unsup_loss = self._compute_binary_unsup_loss(output_ulb_s, pseudo_label, pseudo_mask)

        self._optimize(sup_loss + unsup_loss + feature_loss_weight * feature_loss)

    def _run_step_cardiac(self):
        feature_loss_weight = float(self.cfg["train"]["feature_loss_weight"])

        lb_x, lb_y, ulb_w, ulb_s = self._next_train_batches()
        lb_x = lb_x.to(self.device)
        lb_y = lb_y.to(self.device)
        ulb_w = ulb_w.to(self.device)
        ulb_s = ulb_s.to(self.device)

        output_lb_x, output_ulb_s, feature_loss = self._forward_with_feature_loss(lb_x, ulb_s)

        lb_one_hot = F.one_hot(lb_y.long(), num_classes=output_lb_x.shape[1]).permute(0, 3, 1, 2).float().to(self.device)
        sup_loss = DiceCELoss(softmax=True)(output_lb_x, lb_one_hot)

        pseudo_class, pseudo_one_hot, pseudo_mask = self._build_multiclass_pseudo(output_lb_x, ulb_w)

        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        masked_dice_loss_fn = MaskedDiceLoss(softmax=True)
        unsup_loss = (ce_loss_fn(output_ulb_s, pseudo_class) * pseudo_mask).mean()
        unsup_loss += masked_dice_loss_fn(output_ulb_s, pseudo_one_hot, pseudo_mask.unsqueeze(1))

        self._optimize(sup_loss + unsup_loss + feature_loss_weight * feature_loss)

    def _run_step_spine(self):
        self._run_step_binary(lambda y: y.permute(0, 3, 1, 2).float())

    def _run_step_bladder(self):
        self._run_step_binary(lambda y: y.unsqueeze(1).float())

    def _run_step_colon(self):
        self._run_step_binary(lambda y: (y > 0).unsqueeze(1).float())

    @property
    def is_fully_supervised(self):
        return self._is_fully_supervised

    @is_fully_supervised.setter
    def is_fully_supervised(self, value):
        self._is_fully_supervised = value

    @property
    def train_data_num(self):
        return self._train_data_num

    @property
    def ga_value(self):
        for hook in self._hooks:
            if isinstance(hook, hooks.GA):
                return hook.ga_value
        raise AttributeError("GA hook is not registered")
