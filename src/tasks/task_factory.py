from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from src.evaluation.evaluation_strategy import EvaluationStrategy

class TaskFactory(ABC):
    
    @abstractmethod
    def create_dataset(self, mode, cfg, **kwargs) -> Dataset:
        pass
    
    @abstractmethod
    def create_evaluation_strategy(self, cfg, **kwargs) -> EvaluationStrategy:
        pass