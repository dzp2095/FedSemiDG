from typing import Dict, Type

from src.tasks.bladder_task_factory import BladderTaskFactory
from src.tasks.cardiac_task_factory import CardiacTaskFactory
from src.tasks.colon_task_factory import ColonTaskFactory
from src.tasks.spine_task_factory import SpineTaskFactory
from src.tasks.task_factory import TaskFactory


class TaskRegistry:
    _registry: Dict[str, Type[TaskFactory]] = {}

    @classmethod
    def register_task_factory(cls, task_type: str, factory: Type[TaskFactory]) -> None:
        cls._registry[task_type] = factory

    @classmethod
    def get_factory(cls, task_type: str) -> TaskFactory:
        if task_type in cls._registry:
            return cls._registry[task_type]()
        raise ValueError(f"Unsupported task type: {task_type}")


TaskRegistry.register_task_factory("cardiac", CardiacTaskFactory)
TaskRegistry.register_task_factory("spine", SpineTaskFactory)
TaskRegistry.register_task_factory("bladder", BladderTaskFactory)
TaskRegistry.register_task_factory("colon", ColonTaskFactory)
