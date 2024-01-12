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
        if (
            self.model_id == "stabilityai/sdxl-turbo"
            or self.model_id == "stabilityai/sd-turbo"
        ):
            # SD turbo models were trained without guidance_scale so
            # it should be set to 0
            kwargs["guidance_scale"] = 0.0

            if "num_inference_steps" not in kwargs:
                kwargs["num_inference_steps"] = 1

        return self.ldm(prompt, **kwargs).images

    def __str__(self) -> str:
        return f"TextToImagePipeline model_id={self.model_id}"
