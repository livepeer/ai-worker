from app.pipelines import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir

from diffusers import AutoPipelineForText2Image
import torch
import PIL
from typing import List


class TextToImagePipeline(Pipeline):
    def __init__(self, model_id: str):
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        if torch_device != "cpu":
            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        self.model_id = model_id
        self.ldm = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
        self.ldm.to(get_torch_device())

    def __call__(self, prompt: str, **kwargs) -> List[PIL.Image]:
        return self.ldm(prompt, **kwargs)

    def __str__(self) -> str:
        return f"TextToImagePipeline model_id={self.model_id}"
