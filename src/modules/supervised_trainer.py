import logging
import torch 
import numpy as np
import copy
from tqdm import tqdm

from src.modules.defaults import TrainerBase
from src.modules import hooks
from src.model.unet import UNet
from src.tasks.task_registry import TaskRegistry
from src.datasets.sampler import TrainingSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.losses import DiceCELoss
import torch.nn.functional as F

# trainer for clients have both labeled and unlabeled data
class SupervisedTrainer(TrainerBase):

    def __init__(self, args, cfg) -> None:
        super().__init__(args, cfg)
        self.register_hooks(self.build_hooks())
        task = cfg['task']
        if task == 'fundus':
            self.run_step = self._run_step_fundus
        elif task == 'prostate':
            self.run_step = self._run_step_prostate
        elif task == 'cardiac':
            self.run_step = self._run_step_cardiac
            
    def build_model(self):
        num_channels = self.cfg["model"]["num_channels"]
        num_classes = self.cfg["model"]["num_classes"]
        self.model = UNet(num_channels, num_classes)
    
    def init_dataloader(self):
        batch_size = self.cfg["train"]["batch_size"]
        factory = TaskRegistry.get_factory(self.cfg['task'])
        self.cfg['dataset']['train'] = f"{self.cfg['dataset']['train']}/labeled.csv"
        dataset = factory.create_dataset(mode='train', is_labeled = True, cfg=self.cfg)

        self._train_data_num = len(dataset)
        self.iter_per_epoch = self._train_data_num // batch_size + 1

        batch_size = self.cfg["train"]["batch_size"] * 2
        num_workers = self.cfg["train"]["num_workers"]
        seed = self.cfg["dataset"]["seed"]
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                                                    sampler = TrainingSampler(len(dataset),seed=seed))
        
        self._data_iter = iter(data_loader)

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
        return ret
    
    def before_train(self):
        return super().before_train()

    def after_train(self):
        return super().after_train()
    
    def run_step(self):
        raise NotImplementedError

    def _run_step_fundus(self):
        self.model.train()
        self.model.to(self.device)
        _, image, mask = next(self._data_iter)
        image = image.to(self.device)
        mask = mask.to(self.device)

        cup_mask = mask.eq(0).float()
        disc_mask = mask.le(128).float()
        mask = torch.cat((cup_mask.unsqueeze(1), disc_mask.unsqueeze(1)),dim=1)

        output = self.model(image)
        loss_fn = DiceCELoss(sigmoid=True)

        loss = loss_fn(output, mask)
        
        self.loss_logger.update(loss=loss)
        self.metric_logger.update(loss=loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _run_step_prostate(self):
        self.model.train()
        self.model.to(self.device)
        _, image, mask = next(self._data_iter)
        image = image.to(self.device)
        mask = mask.to(self.device)
        # 0 is the label for the background
        mask = mask.ne(0).long().unsqueeze(1)

        output = self.model(image)
        loss_fn = DiceCELoss(sigmoid=True)

        loss = loss_fn(output, mask)
        
        self.loss_logger.update(loss=loss)
        self.metric_logger.update(loss=loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _run_step_cardiac(self):
        self.model.train()
        self.model.to(self.device)
        _, image, mask = next(self._data_iter)
        image = image.to(self.device)
        mask = mask.to(self.device).long()
        output = self.model(image)
        
        mask = F.one_hot(mask, num_classes=output.shape[1]).permute(0, 3, 1, 2).float().to(device=self.device)

        loss_fn = DiceCELoss(softmax=True)
        loss = loss_fn(output, mask)
        
        self.loss_logger.update(loss=loss)
        self.metric_logger.update(loss=loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @property
    def train_data_num(self):
        return self._train_data_num
    
    @property
    def loss_fn(self):
        return self._loss_fn
    
