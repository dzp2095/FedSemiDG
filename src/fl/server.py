import copy
import logging
import os
import weakref
from typing import List

import numpy as np
import torch

from src.fl.client import Client
from src.utils.metric_logger import MetricLogger

try:
    import wandb
except Exception:
    wandb = None


class Server:
    def __init__(self, clients: List[Client], unseen_client, cfg):
        self.clients = clients
        self.unseen_client = unseen_client
        self.cfg = copy.deepcopy(cfg)

        for client in self.clients:
            client.server = weakref.proxy(self)

        self.r = 0
        self.rounds = int(self.cfg["fl"]["rounds"])
        self.warm_up = int(self.cfg["fl"].get("warm_up", 0))
        self.metric_logger = MetricLogger()

        model_cfg = self.cfg["fl"].get("model", {})
        self.model_save_start_round = int(model_cfg.get("save_start_round", self.rounds + 1))
        self.model_save_interval = int(model_cfg.get("save_interval", 0))

        self.ga_weights_ratio = None
        self.experiment = None
        if self.cfg["fl"].get("wandb_global", False):
            self._wandb_init()

        self.load_model()

    def _wandb_init(self):
        if wandb is None:
            logging.warning("wandb_global is enabled but wandb is not installed; disabled.")
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
        self.experiment.config.update(
            {
                "rounds": self.rounds,
                "local_iter": int(self.cfg["fl"].get("local_iter", 0)),
                "lr": float(self.cfg["train"]["optimizer"]["lr"]),
            },
            allow_val_change=True,
        )

    def _wandb_upload(self):
        if self.experiment:
            self.experiment.log(dict(self.metric_logger._dict))

    def aggregate(self, clients, weights_ratio):
        if not clients:
            raise ValueError("No clients to aggregate")

        w_avg = copy.deepcopy(clients[0].model.state_dict())
        for key in w_avg.keys():
            w_avg[key] = w_avg[key].to("cpu") * weights_ratio[0]

        for idx in range(1, len(clients)):
            state_dict = {k: v.to("cpu") for k, v in clients[idx].model.state_dict().items()}
            for key in w_avg.keys():
                w_avg[key] += state_dict[key] * weights_ratio[idx]
        return w_avg

    def load_model(self):
        resume_path = self.cfg["train"].get("resume_path")
        if resume_path and os.path.isfile(resume_path):
            logging.info("Resume from: %s", resume_path)
            weights = torch.load(resume_path)
            for client in self.clients:
                client.load_model(weights)

    def save_model(self, weights, name):
        output_path = os.path.join(self.cfg["train"]["checkpoint_dir"], f"{name}.pth")
        torch.save(weights, output_path)

    def _weight_clip(self, weight_list):
        clipped = [np.clip(w, 0.0, 1.0) for w in weight_list]
        total = float(sum(clipped))
        if total <= 0:
            return [1.0 / len(clipped)] * len(clipped)
        return [float(w / total) for w in clipped]

    def _refine_weight_by_ga(self, ga_values, initial_step_size=0.05):
        ga_values = np.array(ga_values, dtype=np.float64)
        if np.allclose(np.abs(ga_values).max(), 0.0):
            return

        step = (1.0 / 3.0) * float(initial_step_size)
        norm_gap = ga_values / np.max(np.abs(ga_values))

        decay = 1.0 - (float(self.r) / float(max(1, self.rounds)))
        step = step * decay

        for idx, gap in enumerate(norm_gap):
            self.ga_weights_ratio[idx] += float(gap) * step

        self.ga_weights_ratio = self._weight_clip(self.ga_weights_ratio)

    def _compute_weights_ratio(self, runned_clients):
        ga_values = []
        ga_ready = True
        for client in runned_clients:
            try:
                ga_values.append(float(client.trainer.ga_value))
            except Exception:
                ga_ready = False
                break

        if ga_ready and len(ga_values) == len(runned_clients):
            if self.ga_weights_ratio is None:
                num_clients = len(runned_clients)
                self.ga_weights_ratio = [1.0 / num_clients] * num_clients
                weights_ratio = list(self.ga_weights_ratio)
                self._refine_weight_by_ga(ga_values)
            else:
                self._refine_weight_by_ga(ga_values)
                weights_ratio = list(self.ga_weights_ratio)

            logging.info("Round %d GAA weights ratio: %s", self.r, weights_ratio)
            return weights_ratio

        train_nums = [client.train_data_num for client in runned_clients]
        train_sum = float(sum(train_nums))
        if train_sum <= 0:
            return [1.0 / len(runned_clients)] * len(runned_clients)

        weights_ratio = [float(num) / train_sum for num in train_nums]
        logging.info("Round %d data-size weights ratio: %s", self.r, weights_ratio)
        return weights_ratio

    def run(self):
        for self.r in range(self.rounds):
            runned_clients = self.run_clients()
            weights_ratio = self._compute_weights_ratio(runned_clients)

            weights = self.aggregate(runned_clients, weights_ratio)
            self.distribute_global_model(weights)
            self.log_round_metrics(runned_clients)
            self.maybe_save_model(weights)
            self._wandb_upload()

        final_weights = self.clients[0].model.state_dict()
        self.save_model(final_weights, "global_model_final")
        if self.experiment:
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

    def distribute_global_model(self, weights):
        for client in self.clients:
            client.load_model(weights)

    def log_round_metrics(self, clients):
        losses = []
        for client in clients:
            loss = client.trainer.metric_logger._dict.get("loss")
            if isinstance(loss, (float, int)):
                losses.append(float(loss))

        if losses:
            avg_loss = sum(losses) / len(losses)
            self.metric_logger.update(round=float(self.r), train_loss=avg_loss)
            logging.info("Round %d average client loss: %.6f", self.r, avg_loss)

    def maybe_save_model(self, weights):
        if self.model_save_interval <= 0:
            return
        if self.r < self.model_save_start_round:
            return
        if (self.r - self.model_save_start_round) % self.model_save_interval != 0:
            return

        self.save_model(weights, f"global_model_round_{self.r}")
