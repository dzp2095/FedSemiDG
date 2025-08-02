import os
import logging
import torch
import copy
import glob
import weakref
from typing import List
import wandb
import numpy as np

from src.utils.device_selector import get_free_device_name
from src.utils.metric_logger import MetricLogger
from src.fl.client import Client
from src.tasks.task_registry import TaskRegistry

class Server:
    def __init__(self, clients: List[Client], unseen_client, cfg):
        self.clients = clients
        self.unseen_client = unseen_client
        self.r = 0
        for client in self.clients:
            client.server = weakref.proxy(self)
        self.factory = TaskRegistry.get_factory(cfg['task'])
        self.evaluation_strategy = self.factory.create_evaluation_strategy(cfg)

        self.save_checkpoints = True  # turn on to save the checkpoints
        self.rounds = cfg['fl']['rounds']
        self.warm_up = cfg['fl']['warm_up']

        # weight ratio is the same for all clients, as the iteration number is the same
        self.weights_ratio = [1.0 / len(self.clients)] * len(self.clients) 
        
        self.test_start_round = cfg['fl']['test_start_round']
        self.model_save_start_round = cfg['fl']['model']['save_start_round']
        self.model_save_interval = cfg['fl']['model']['save_interval']
        self.result_save_start_round = cfg['fl']['result']['save_start_round']
        self.result_save_interval = cfg['fl']['result']['save_interval']

        self.metric_logger = MetricLogger()
        self.cfg = copy.deepcopy(cfg)
        self.device = get_free_device_name()

        if self.cfg["fl"]["wandb_global"]:
            self.wandb_init()
        self.load_model()
       
    def wandb_init(self):
        wandb.login(key=self.cfg["wandb"]["key"])
        self.wandb_id = wandb.util.generate_id()
        self.experiment = wandb.init(project=f'{self.cfg["wandb"]["project"]}', resume='allow', id=self.wandb_id, name=self.cfg["wandb"]["run_name"])
        self.experiment.config.update(
            dict(steps=self.cfg["train"]["max_iter"], batch_size=  self.cfg["train"]["batch_size"],
                 learning_rate = self.cfg["train"]["optimizer"]["lr"]), allow_val_change=True)
       
        logging.info("######## Running wandb logger")

    def wandb_upload(self, metric_logger):
        self.experiment.log(metric_logger._dict)

    def aggregate(self, clients, weights_ratio):
        # FedAvg
        w_avg = copy.deepcopy(clients[0].model.state_dict())
        for k in w_avg.keys():
            w_avg[k] = w_avg[k].to('cpu') * weights_ratio[0]

        for i in range(1, len(clients)):
            client_state_dict = {k: v.to('cpu') for k, v in clients[i].model.state_dict().items()}
            for k in w_avg.keys():
                w_avg[k] += client_state_dict[k] * weights_ratio[i]
        return w_avg

    def load_model(self):
        resume_path = self.cfg['train']['resume_path']
        if resume_path is not None and os.path.isfile(resume_path):
            logging.info(f"Resume from: {resume_path}")
            w = torch.load(resume_path)
            # send global model
            for client in self.clients:
                client.load_model(w)

    def save_model(self, w_avg, name):
        torch.save(w_avg,
            os.path.join(self.cfg["train"]["checkpoint_dir"], name + '.pth')
        )

    # reference from: https://github.com/MediaBrain-SJTU/FedDG-GA
    def refine_weight_by_GA(self, ga_values, initial_step_size=0.05):
        ga_values = np.array(ga_values)
        initial_step_size = 1./3. * initial_step_size
        norm_gap_list = ga_values / np.max(np.abs(ga_values))

        # Linear decay
        current_round = self.r
        total_rounds = self.rounds
        decayed_step_size = initial_step_size * (1 - current_round / total_rounds)

        for i, norm_gap in enumerate(norm_gap_list):
            self.ga_weights_ratio[i] += norm_gap * decayed_step_size

        self.ga_weights_ratio = self.weight_clip(self.ga_weights_ratio)
        
    def weight_clip(self, weight_list):
        # Clip weights
        clipped_weights = [np.clip(w, 0.0, 1.0) for w in weight_list]
        # Sum of weights
        total_weight = sum(clipped_weights)
        # Normalize
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in clipped_weights]
        else:
            normalized_weights = [1.0 / len(clipped_weights)] * len(clipped_weights)
        return normalized_weights

    def run(self):
        self.best_metric = 0
        self.ga_weights_ratio = None

        for self.r in range(self.rounds):
            runned_clients = self.run_clients()
            if self.cfg["fl"]["use_ga"]==True:
                ga_values = [client.trainer.ga_value for client in runned_clients]
                if self.ga_weights_ratio is None:
                    # initialize the weights
                    num_client = len(runned_clients)
                    self.ga_weights_ratio = [1.0 / num_client] * num_client
                    weights_ratio = self.ga_weights_ratio
                    # update the weights, but don't use the updated weights for the first time
                    self.refine_weight_by_GA(ga_values)
                else:
                    self.refine_weight_by_GA(ga_values)
                    weights_ratio = self.ga_weights_ratio
                logging.info(f"######## GA weights ratio: {weights_ratio}")
            else:
                train_nums = [client.train_data_num for client in runned_clients]
                train_num_sum = sum(train_nums)
                weights_ratio = [num / train_num_sum for num in train_nums]
                logging.info(f"######## Weights ratio: {weights_ratio}")
            w_avg = self.aggregate(runned_clients, weights_ratio)
            self.distribute_global_model(w_avg)
            self.run_evaluation(w_avg)
        
        if self.cfg["fl"]["wandb_global"]:
            self.experiment.finish()

    def run_clients(self):
        runned_clients = []
        if self.r < self.warm_up:
            for client in self.clients:
                if client.is_labeled_client:
                    client.run()
                    runned_clients.append(client)
        else:
            for client in self.clients:
                client.run()
                runned_clients.append(client)
        return runned_clients

        
    
    def distribute_global_model(self, w_avg):
        for client in self.clients:
            client.load_model(w_avg)
        
    def run_evaluation(self, w_avg):
        # save the global model automatically
        if (self.r >= self.model_save_start_round and (self.r - self.model_save_start_round) % self.model_save_interval == 0):
            self.save_model(w_avg, f"global_model_round_{self.r}")

        if (self.r >= self.test_start_round):
            self.global_test(w_avg)
        
        # upload the metrics to wandb
        if self.cfg["fl"]["wandb_global"]:
            self.wandb_upload(self.metric_logger)
    
    def global_test(self, w_avg):
        model = copy.deepcopy(self.clients[0].model)
        model.load_state_dict(w_avg)
        
        root_dir = self.cfg['dataset']['test']
        cfg = copy.deepcopy(self.cfg)
       
        test_csv = os.path.join(root_dir, self.unseen_client, 'all.csv')
        cfg['dataset']['test'] = test_csv
        dataset = self.factory.create_dataset(mode='test', cfg=cfg)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg["train"]["batch_size"], shuffle=False, 
                                                num_workers=8, pin_memory=True)
        
        if self.r >= self.result_save_start_round and (self.r - self.result_save_start_round) % self.result_save_interval == 0:
            save_path = self.cfg['wandb']['run_name'] + f"_round_{self.r}_{self.unseen_client}"
        else:
            save_path = None
        metrics = self.evaluation_strategy.test(model, data_loader, self.device, save_path)
        metrics = {f"{self.unseen_client}/{key}": value for key, value in metrics.items()}
        self.metric_logger.update(**metrics)
        logging.info(f"######## Global test on {self.unseen_client} : {metrics}")