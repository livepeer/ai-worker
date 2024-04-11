from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir

from diffusers import DiffusionPipeline
from huggingface_hub import file_download
import torch
import PIL
from typing import List
import logging
import os

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class TextToVideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        has_fp16_variant = any(
            ".fp16.safetensors" in fname
            for _, _, files in os.walk(folder_path)
            for fname in files
        )
        if torch_device != "cpu" and has_fp16_variant:
            logger.info("TextToVideoPipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        self.model_id = model_id
        self.ldm = DiffusionPipeline.from_pretrained(model_id, **kwargs)
        self.ldm.to(get_torch_device())

        if os.environ.get("SFAST"):
            logger.info(
                "TextToVideoPipeline will be dynamicallly compiled with stable-fast for %s",
                model_id,
            )
            from app.pipelines.sfast import compile_model

            self.ldm = compile_model(self.ldm)

    def __call__(self, prompt: str, **kwargs) -> List[List[PIL.Image]]:
        if "decode_chunk_size" not in kwargs:
            kwargs["decode_chunk_size"] = 4

        # ali-vilab/text-to-video-ms-1.7b has a limited parameter set
        if (
            self.model_id == "ali-vilab/text-to-video-ms-1.7b"
        ):
            if "fps" in kwargs:
                del kwargs["fps"]
            if "MotionBucketId" in kwargs:
                del kwargs["MotionBucketId"]
            if "NoiseAugStrength" in kwargs:
                del kwargs["NoiseAugStrength"]

        seed = kwargs.pop("seed", None)
        if seed is not None:
            if isinstance(seed, int):
                kwargs["generator"] = torch.Generator(get_torch_device()).manual_seed(
                    seed
                )
            elif isinstance(seed, list):
                kwargs["generator"] = [
                    torch.Generator(get_torch_device()).manual_seed(s) for s in seed
                ]

        return self.ldm(prompt, **kwargs).frames

    def __str__(self) -> str:
        return f"TextToVideoPipeline model_id={self.model_id}"
