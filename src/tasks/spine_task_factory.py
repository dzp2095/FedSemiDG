from src.datasets.dataset_spine import SpineDataset
from src.evaluation.spine_eval import SpineEvalStrategy
from src.tasks.task_factory import TaskFactory


class SpineTaskFactory(TaskFactory):
    def create_evaluation_strategy(self, cfg, **kwargs):
        return SpineEvalStrategy(cfg)

    def create_dataset(self, mode, cfg, **kwargs):
        is_labeled = kwargs.get("is_labeled", False)
        return SpineDataset(mode=mode, cfg=cfg, is_labeled=is_labeled)
