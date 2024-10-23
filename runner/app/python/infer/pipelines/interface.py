from abc import ABC, abstractmethod
from PIL import Image

class Pipeline(ABC):
    def __init__(self, **params):
        pass

    @abstractmethod
    def process_frame(self, frame: Image.Image) -> Image.Image:
        pass

    @abstractmethod
    def update_params(self, **params):
        pass
