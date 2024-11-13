from abc import ABC, abstractmethod
from typing import Any


class Pipeline(ABC):
    @abstractmethod
    def __init__(self, model_id: str, model_dir: str):
        self.model_id: str # type hint so we can use the field in routes
        raise NotImplementedError("Pipeline should implement an __init__ method")

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        raise NotImplementedError("Pipeline should implement a __call__ method")
