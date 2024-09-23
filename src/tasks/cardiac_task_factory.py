import torch.nn as nn

from src.evaluation.cardiac_eval import CardiacEvalStrategy
from src.tasks.task_factory import TaskFactory
from src.datasets.dataset_cardiac import CardiacDataset

class CardiacTaskFactory(TaskFactory):
    def create_evaluation_strategy(self, cfg, **kwargs):
        return CardiacEvalStrategy(cfg)
    
    def create_dataset(self, mode, cfg, **kwargs):
        is_labeled = kwargs.get('is_labeled', False)
        return CardiacDataset(mode=mode, cfg=cfg, is_labeled=is_labeled)
