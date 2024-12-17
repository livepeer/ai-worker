from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

class HealthCheck(BaseModel):
    status: str = "OK"

class Pipeline(ABC):
    @abstractmethod
    def __init__(self, model_id: str, model_dir: str):
        self.model_id: str # declare the field here so the type hint is available when using this abstract class
        raise NotImplementedError("Pipeline should implement an __init__ method")

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        raise NotImplementedError("Pipeline should implement a __call__ method")

    def get_status(self) -> HealthCheck:
        """
        Returns a status object for the pipeline.
        """
        return HealthCheck(status="OK")
