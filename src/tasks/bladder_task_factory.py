from src.datasets.dataset_bladder import BladderDataset
from src.evaluation.bladder_eval import BladderEvalStrategy
from src.tasks.task_factory import TaskFactory


class BladderTaskFactory(TaskFactory):
    def create_evaluation_strategy(self, cfg, **kwargs):
        return BladderEvalStrategy(cfg)

    def create_dataset(self, mode, cfg, **kwargs):
        is_labeled = kwargs.get("is_labeled", False)
        return BladderDataset(mode=mode, cfg=cfg, is_labeled=is_labeled)
