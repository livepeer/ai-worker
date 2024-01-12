from app.pipelines import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir

from diffusers import StableVideoDiffusionPipeline
import torch
import PIL
from typing import List


class ImageToVideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        if torch_device != "cpu":
            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        self.model_id = model_id
        self.ldm = StableVideoDiffusionPipeline.from_pretrained(model_id, **kwargs)
        self.ldm.to(get_torch_device())

    def __call__(self, image: PIL.Image, **kwargs) -> List[List[PIL.Image]]:
        if "decode_chunk_size" not in kwargs:
            kwargs["decode_chunk_size"] = 8

        return self.ldm(image, **kwargs).frames

    def __str__(self) -> str:
        return f"ImageToVideoPipeline model_id={self.model_id}"
