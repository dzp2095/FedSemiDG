from src.datasets.dataset_colon import ColonDataset
from src.evaluation.colon_eval import ColonEvalStrategy
from src.tasks.task_factory import TaskFactory


class ColonTaskFactory(TaskFactory):
    def create_evaluation_strategy(self, cfg, **kwargs):
        return ColonEvalStrategy(cfg)

    def create_dataset(self, mode, cfg, **kwargs):
        is_labeled = kwargs.get("is_labeled", False)
        return ColonDataset(mode=mode, cfg=cfg, is_labeled=is_labeled)
