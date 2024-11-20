
import logging
import torch 
import wandb
import copy
import os
import glob
from datetime import datetime

from src.modules.defaults import HookBase
from src.utils.device_selector import get_free_device_name
from src.tasks.task_registry import TaskRegistry
from src.model.ema import ModelEMA
from torch.utils.data import ConcatDataset

class Timer(HookBase):
    def before_train(self):
        self.tick = datetime.now()
        logging.info("Begin training at: {}".format(self.tick.strftime("%Y-%m-%d %H:%M:%S")))
        logging.info("######## Running Timer")

    def after_train(self):
        tock = datetime.now()
        logging.info("\nBegin training at: {}".format(self.tick.strftime("%Y-%m-%d %H:%M:%S")))
        logging.info("Finish training at: {}".format(tock.strftime("%Y-%m-%d %H:%M:%S")))
        logging.info("Time spent: {}\n".format(str(tock - self.tick).split('.')[0]))

class WAndBUploader(HookBase):
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        wandb.login(key=self.cfg["wandb"]["key"])
        self.wandb_id = wandb.util.generate_id()

    def before_train(self):
        self.experiment = wandb.init(project=f'{self.cfg["wandb"]["project"]}', resume='allow', id=self.wandb_id, name=self.cfg["wandb"]["run_name"])
        self.val_interval = self.cfg["train"]["eval_interval"]
        self.experiment.config.update(
            dict(steps=self.trainer.max_iter, batch_size=self.cfg["train"]["batch_size"],
                 learning_rate = self.cfg["train"]["optimizer"]["lr"]), allow_val_change=True)
       
        logging.info("######## Running wandb logger")

    def after_step(self):
        wandb_dict = {}
        metric_names = {"dc", "jc", "hd", "asd"}
        for metric_name in metric_names:
            if metric_name in self.trainer.metric_logger._dict:
                # get the current value of the metric
                current_value = self.trainer.metric_logger._dict[metric_name]

                # only log the metric if it has changed
                if not hasattr(self, f'prev_{metric_name}') or current_value != getattr(self, f'prev_{metric_name}'):
                    wandb_dict.update({metric_name: current_value})
                    setattr(self, f'prev_{metric_name}', current_value)

        if self.trainer.iter % self.val_interval == 0 and wandb_dict:
            wandb_dict.update(self.trainer.metric_logger._dict)
            self.experiment.log(wandb_dict)

    def after_train(self):
        self.experiment.finish()

class EvalHook(HookBase):
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.factory = TaskRegistry.get_factory(cfg['task'])
        self.evaluation_strategy = self.factory.create_evaluation_strategy(cfg)
        
    # test on the new global model in FL mode
    def before_step(self):
        eval_interval = self.cfg["local"]["eval_interval"]
        eval_start = self.cfg['local']['eval_start']
        if self.trainer.iter >= eval_start and eval_interval > 0 and (self.trainer.iter - eval_start) % eval_interval == 0:
            root_dir = self.cfg['dataset']['test']
            cfg = copy.deepcopy(self.cfg)
            client_folders = [os.path.basename(f) for f in glob.glob(os.path.join(root_dir, 'client*')) if os.path.isdir(f)]
            for client_folder in client_folders:
                test_csv = os.path.join(root_dir, client_folder, 'all.csv')
                cfg['dataset']['test'] = test_csv
                dataset = self.factory.create_dataset(mode='test', cfg=cfg)
                data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg["train"]["batch_size"], shuffle=False, 
                                                        num_workers=8, pin_memory=True)
                device = get_free_device_name(gpu_exclude_list=self.cfg["train"]["gpu_exclude"])
                model = self.trainer.model
                
                save_start = self.cfg['local']['save_start']
                save_interval = self.cfg['local']['save_interval']
                if self.trainer.iter >= save_start and (self.trainer.iter - save_start) % save_interval == 0:
                    save_path = self.cfg['wandb']['run_name'] + f'_iter_{self.trainer.iter}'
                else:
                    save_path = None
                metrics = self.evaluation_strategy.test(model, data_loader, device, save_path)
                metrics = {f"{client_folder}/{key}": value for key, value in metrics.items()}
                self.trainer.metric_logger.update(**metrics)
                logging.info(f"######## Locailized test on {client_folder} : {metrics}")
        return super().before_step()

class GA(HookBase):
    def __init__(self, cfg):
        self._ga_value = 1
        self.cfg = copy.deepcopy(cfg)
        self.factory = TaskRegistry.get_factory(cfg['task'])
        self.evaluation_strategy = self.factory.create_evaluation_strategy(cfg)

    def before_train(self):
        # save the global model
        self.global_model = copy.deepcopy(self.trainer.model)
        return super().before_train()

    def after_train(self):
        # evaluation of the updated local model and the global model on the training set
        train_root = self.cfg['dataset']['train']
        self.cfg['dataset']['only_image'] = os.path.join(train_root, 'labeled.csv')
        labeled_dataset = self.factory.create_dataset(mode='only_image', cfg=self.cfg)
        self.cfg['dataset']['only_image'] = os.path.join(train_root, 'unlabeled.csv')
        unlabeled_dataset = self.factory.create_dataset(mode='only_image', cfg=self.cfg)
        self.cfg['dataset']['only_image'] = train_root
        dataset = ConcatDataset([labeled_dataset, unlabeled_dataset])
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg["train"]["batch_size"], shuffle=False, 
                                        num_workers=8, pin_memory=True)
        self.ga_value = self.evaluation_strategy.cal_kl_loss(self.trainer.model, self.global_model, data_loader, self.trainer.device)
        return super().after_train()
    
    @property
    def ga_value(self):
        return self._ga_value
    
    @ga_value.setter
    def ga_value(self, value):
        self._ga_value = value

class EMA(HookBase):
    def __init__(self, cfg):
        self.decay = cfg["train"]["ema_decay"]

    def before_train(self):
        if self.trainer.local_teacher is None:
            self.trainer.local_teacher = ModelEMA(self.trainer.device, self.trainer.model, self.decay)

    def after_step(self):
        self.trainer.local_teacher.update(self.trainer.model)