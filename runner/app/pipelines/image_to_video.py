from app.pipelines import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir

from diffusers import StableVideoDiffusionPipeline
from huggingface_hub import model_info
import torch
import PIL
from typing import List
import logging

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class ImageToVideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        model_data = model_info(model_id)
        has_fp16_variant = any(
            ".fp16.safetensors" in file.rfilename for file in model_data.siblings
        )
        if torch_device != "cpu" and has_fp16_variant:
            logger.info("ImageToVideoPipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        self.model_id = model_id
        self.ldm = StableVideoDiffusionPipeline.from_pretrained(model_id, **kwargs)
        self.ldm.to(get_torch_device())

    def __call__(self, image: PIL.Image, **kwargs) -> List[List[PIL.Image]]:
        if "decode_chunk_size" not in kwargs:
            kwargs["decode_chunk_size"] = 8

        seed = kwargs.pop("seed", None)
        if seed is not None:
            kwargs["generator"] = torch.Generator(seed)

        return self.ldm(image, **kwargs).frames

    def __str__(self) -> str:
        return f"ImageToVideoPipeline model_id={self.model_id}"
