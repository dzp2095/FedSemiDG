import logging
import torch 
import numpy as np
import copy
from tqdm import tqdm
import torch.nn.functional as F
import os

from src.modules.defaults import TrainerBase
from src.modules import hooks
from src.model.unet import UNet
from src.tasks.task_registry import TaskRegistry
from src.datasets.sampler import TrainingSampler
from monai.losses import DiceCELoss, MaskedDiceLoss

# trainer for clients have both labeled and unlabeled data
class SemiTrainer(TrainerBase):

    def __init__(self, args, cfg, is_fully_supervised=True) -> None:
        super().__init__(args, cfg)
        self.register_hooks(self.build_hooks())
        self._is_fully_supervised = is_fully_supervised
        task = cfg['task']
        if task == 'fundus':
            self.run_step = self._run_step_fundus
        elif task == 'prostate':
            self.run_step = self._run_step_prostate
        elif task == 'cardiac':
            self.run_step = self._run_step_cardiac

        self.ema_model = None
            
    def build_model(self):
        num_channels = self.cfg["model"]["num_channels"]
        num_classes = self.cfg["model"]["num_classes"]
        self.model = UNet(num_channels, num_classes)
    
    def init_dataloader(self):
        batch_size = self.cfg["train"]["batch_size"]
        factory = TaskRegistry.get_factory(self.cfg['task'])
        train_root = self.cfg['dataset']['train']
        self.cfg['dataset']['train'] = os.path.join(train_root, 'labeled.csv')
        labeled_dataset = factory.create_dataset(mode='train', is_labeled = True, cfg=self.cfg)
        self.cfg['dataset']['train'] = os.path.join(train_root, 'unlabeled.csv')
        unlabeled_dataset = factory.create_dataset(mode='train', is_labeled = False, cfg=self.cfg)
        self.labeled_data_num = len(labeled_dataset)
        self.unlabeled_data_num = len(unlabeled_dataset)

        self._train_data_num = self.labeled_data_num + self.unlabeled_data_num
        self.iter_per_epoch = self._train_data_num // batch_size + 1

        batch_size = self.cfg["train"]["batch_size"]
        num_workers = self.cfg["train"]["num_workers"]
        seed = self.cfg["dataset"]["seed"]
        labeled_data_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                                                    sampler = TrainingSampler(len(labeled_dataset),seed=seed))
        unlabeled_data_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                                                    sampler = TrainingSampler(len(unlabeled_dataset),seed=seed))
        
        self._labeled_data_iter = iter(labeled_data_loader)
        self._unlabeled_data_iter = iter(unlabeled_data_loader)

    def load_model(self, model_weights):
        self.model.load_state_dict(model_weights, strict=False)
        # need to construct a new optimizer for the new network
        old_scheduler = copy.deepcopy(self.lr_scheduler.state_dict())
        old_optimizer = copy.deepcopy(self.optimizer.state_dict())
        self.build_optimizer()
        self.optimizer.load_state_dict(old_optimizer)
        self.build_schedular(self.optimizer)
        self.lr_scheduler.load_state_dict(old_scheduler)

    def build_hooks(self):
        ret = [hooks.Timer()]
        if self.cfg["hooks"]["wandb"]:
            ret.append(hooks.WAndBUploader(self.cfg))

        if self.cfg["local"]["eval_interval"]!=0:
            ret.append(hooks.EvalHook(self.cfg))
        ret.append(hooks.EMA(self.cfg))
        return ret
    
    def before_train(self):
        self._train_data_num = 0
        return super().before_train()

    def after_train(self):
        return super().after_train()
    
    def run_step(self):
        raise NotImplementedError

    def _run_step_fundus(self):
        self.model.train()
        self.model.to(self.device)
        threshold = self.cfg['train']['pseudo_label_threshold'] 

        _, lb_x, lb_y = next(self._labeled_data_iter)
        _, ulb_w, ulb_s = next(self._unlabeled_data_iter)
        lb_x, lb_y, ulb_w, ulb_s = lb_x.to(self.device), lb_y.to(self.device), ulb_w.to(self.device), ulb_s.to(self.device)
        lb_y_cup = lb_y.eq(0).float()
        lb_y_disc = lb_y.le(128).float()
        lb_y = torch.cat((lb_y_cup.unsqueeze(1), lb_y_disc.unsqueeze(1)),dim=1)

        # generate pseudo labels
        with torch.no_grad():
            ulb_logit = self.ema_model.ema(ulb_w)
            ulb_prob = ulb_logit.sigmoid()
            ulb_y = ulb_prob.ge(0.5).float()
            ulb_y_mask = ulb_prob.ge(threshold).float() + ulb_prob.lt(1-threshold).float()

        data = torch.cat((lb_x, ulb_s), dim=0)
        lb_sz = lb_x.size(0)
        output = self.model(data)
        output_lb_x, output_ulb_s = output[:lb_sz], output[lb_sz:]
        # calculate supervised loss
        dice_ce_loss_fn = DiceCELoss(sigmoid=True)
        sup_loss = dice_ce_loss_fn(output_lb_x, lb_y)
        # calculate consistency loss
        ce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        masked_dice_loss_fn = MaskedDiceLoss(sigmoid=True)
        unsup_loss = (ce_loss_fn(output_ulb_s, ulb_y)*ulb_y_mask).mean()
        # calculate masked dice loss for each class
        for i in range(output_ulb_s.size(1)):             
            unsup_loss += masked_dice_loss_fn(output_ulb_s[:, i, :, :].unsqueeze(1), ulb_y[:, i, :, :].unsqueeze(1), ulb_y_mask[:, i, :, :].unsqueeze(1))
        
        loss = sup_loss + unsup_loss
        self.loss_logger.update(loss=loss)
        self.metric_logger.update(loss=loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _run_step_prostate(self):
        self.model.train()
        self.model.to(self.device)
        threshold = self.cfg['train']['pseudo_label_threshold'] 

        _, lb_x, lb_y = next(self._labeled_data_iter)
        _, ulb_w, ulb_s = next(self._unlabeled_data_iter)
        lb_x, lb_y, ulb_w, ulb_s = lb_x.to(self.device), lb_y.to(self.device), ulb_w.to(self.device), ulb_s.to(self.device)
        # 0 is the label for the background
        lb_y = lb_y.ne(0).long().unsqueeze(1)
        
        # generate pseudo labels
        with torch.no_grad():
            ulb_logit = self.ema_model.ema(ulb_w)
            ulb_prob = ulb_logit.sigmoid()
            ulb_y = ulb_prob.ge(0.5).float()
            ulb_y_mask = ulb_prob.ge(threshold).float() + ulb_prob.lt(1-threshold).float()

        data = torch.cat((lb_x, ulb_s), dim=0)
        lb_sz = lb_x.size(0)
        output = self.model(data)
        output_lb_x, output_ulb_s = output[:lb_sz], output[lb_sz:]
        # calculate supervised loss
        dice_ce_loss_fn = DiceCELoss(sigmoid=True)
        sup_loss = dice_ce_loss_fn(output_lb_x, lb_y)
        # calculate consistency loss
        ce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        masked_dice_loss_fn = MaskedDiceLoss(sigmoid=True)
        unsup_loss = (ce_loss_fn(output_ulb_s, ulb_y)*ulb_y_mask).mean()
        # calculate masked dice loss for each class
        for i in range(output_ulb_s.size(1)):             
            unsup_loss += masked_dice_loss_fn(output_ulb_s[:, i, :, :].unsqueeze(1), ulb_y[:, i, :, :].unsqueeze(1), ulb_y_mask[:, i, :, :].unsqueeze(1))

        loss = sup_loss + unsup_loss
        self.loss_logger.update(loss=loss)
        self.metric_logger.update(loss=loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _run_step_cardiac(self):
        self.model.train()
        self.model.to(self.device)
        threshold = self.cfg['train']['pseudo_label_threshold'] 

        _, lb_x, lb_y = next(self._labeled_data_iter)
        _, ulb_w, ulb_s = next(self._unlabeled_data_iter)
        lb_x, lb_y, ulb_w, ulb_s = lb_x.to(self.device), lb_y.to(self.device), ulb_w.to(self.device), ulb_s.to(self.device)

        data = torch.cat((lb_x, ulb_s), dim=0)
        lb_sz = lb_x.size(0)
        output = self.model(data)

        lb_y = F.one_hot(lb_y.long(), num_classes=output.shape[1]).permute(0, 3, 1, 2).float().to(device=self.device)

        # generate pseudo labels
        with torch.no_grad():
            ulb_logit = self.ema_model.ema(ulb_w)
            ulb_prob = torch.softmax(ulb_logit, dim=1)
            ulb_y_conf, ulb_y = torch.max(ulb_prob, dim=1)
            ulb_y_one_hot = F.one_hot(ulb_y, num_classes=ulb_logit.shape[1]).permute(0, 3, 1, 2).float()
            ulb_y_mask = ulb_y_conf.ge(threshold).float()

        output_lb_x, output_ulb_s = output[:lb_sz], output[lb_sz:]
        # calculate supervised loss
        dice_ce_loss_fn = DiceCELoss(softmax=True)
        sup_loss = dice_ce_loss_fn(output_lb_x, lb_y)
        # calculate consistency loss
        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        masked_dice_loss_fn = MaskedDiceLoss(softmax=True)
        unsup_loss = (ce_loss_fn(output_ulb_s, ulb_y)*ulb_y_mask).mean()
        # calculate masked dice loss for each class    
        unsup_loss += masked_dice_loss_fn(output_ulb_s, ulb_y_one_hot, ulb_y_mask.unsqueeze(1))

        loss = sup_loss + unsup_loss
        self.loss_logger.update(loss=loss)
        self.metric_logger.update(loss=loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @property
    def is_fully_supervised(self):
        return self._is_fully_supervised
    
    @is_fully_supervised.setter
    def is_fully_supervised(self, value):
        self._is_fully_supervised = value

    @property
    def train_data_num(self):
        return self._train_data_num

    
