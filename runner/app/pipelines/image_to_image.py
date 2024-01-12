from app.pipelines import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir

from diffusers import AutoPipelineForImage2Image
import torch
import PIL
from typing import List


class ImageToImagePipeline(Pipeline):
    def __init__(self, model_id: str):
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        if torch_device != "cpu":
            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        self.model_id = model_id
        self.ldm = AutoPipelineForImage2Image.from_pretrained(model_id, **kwargs)
        self.ldm.to(get_torch_device())

    def __call__(self, prompt: str, image: PIL.Image, **kwargs) -> List[PIL.Image]:
        if (
            self.model_id == "stabilityai/sdxl-turbo"
            or self.model_id == "stabilityai/sd-turbo"
        ):
            # SD turbo models were trained without guidance_scale so
            # it should be set to 0
            kwargs["guidance_scale"] = 0.0

            # num_inference_steps * strength should be >= 1 because
            # the pipeline will be run for int(num_inference_steps * strength) steps
            if "strength" not in kwargs:
                kwargs["strength"] = 0.5

            if "num_inference_steps" not in kwargs:
                kwargs["num_inference_steps"] = 2

        return self.ldm(prompt, image=image, **kwargs).images

    def __str__(self) -> str:
        return f"ImageToImagePipeline model_id={self.model_id}"
