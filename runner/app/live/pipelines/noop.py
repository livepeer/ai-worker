from PIL import Image

from .interface import Pipeline

class Noop(Pipeline):
  def __init__(self, **params):
    super().__init__(**params)

  def process_frame(self, image: Image.Image) -> Image.Image:
    return image.convert("RGB")

  def update_params(self, **params):
    pass
