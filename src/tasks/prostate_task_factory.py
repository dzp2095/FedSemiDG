import torch.nn as nn

from src.evaluation.prostate_eval import ProstateEvalStrategy
from src.tasks.task_factory import TaskFactory
from src.datasets.dataset_prostate import ProstateDataset

class ProstateTaskFactory(TaskFactory):
    def create_evaluation_strategy(self, cfg, **kwargs):
        return ProstateEvalStrategy(cfg)
    
    def create_dataset(self, mode, cfg, **kwargs):
        is_labeled = kwargs.get('is_labeled', False)
        return ProstateDataset(mode=mode, cfg=cfg, is_labeled=is_labeled)
