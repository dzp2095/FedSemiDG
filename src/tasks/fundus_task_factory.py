import torch.nn as nn

from src.evaluation.fundus_eval import FundusEvalStrategy
from src.tasks.task_factory import TaskFactory
from src.datasets.dataset_fundus import FundusDataset

class FundusTaskFactory(TaskFactory):
    def create_evaluation_strategy(self, cfg, **kwargs):
        return FundusEvalStrategy(cfg)
    
    def create_dataset(self, mode, cfg, **kwargs):
        is_labeled = kwargs.get('is_labeled', False)
        return FundusDataset(mode=mode, cfg=cfg, is_labeled=is_labeled)