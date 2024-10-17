from typing import Type, Dict

from src.tasks.task_factory import TaskFactory
from src.tasks.fundus_task_factory import FundusTaskFactory
from src.tasks.cardiac_task_factory import CardiacTaskFactory
from src.tasks.prostate_task_factory import ProstateTaskFactory
from src.tasks.spine_task_factory import SpineTaskFactory

class TaskRegistry:
    _registry: Dict[str, Type[TaskFactory]] = {}

    @classmethod
    def register_task_factory(cls, task_type: str, factory: Type[TaskFactory]) -> None:
        cls._registry[task_type] = factory

    @classmethod
    def get_factory(cls, task_type: str) -> TaskFactory:
        if task_type in cls._registry:
            return cls._registry[task_type]()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

# register tasks
TaskRegistry.register_task_factory("fundus", FundusTaskFactory)
TaskRegistry.register_task_factory("prostate", ProstateTaskFactory)
TaskRegistry.register_task_factory("cardiac", CardiacTaskFactory)
TaskRegistry.register_task_factory("spine", SpineTaskFactory)
