from app.pipelines.base import Pipeline

from app.pipelines.utils import (
    SafetyChecker,
    get_model_dir,
    get_torch_device,
    is_lightning_model,
    is_turbo_model,
    split_prompt,
)

from diffusers import CogVideoXPipeline, DiffusionPipeline
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

        # Note: we are forcing a download of recommended model weights here. Probably not what we want.
        if model_id == "THUDM/CogVideoX-2b":
            if "variant" in kwargs:
                del kwargs["variant"]
            logger.info("TextToVideoPipeline loading fp16 variant for %s", model_id)
            kwargs["torch_dtype"] = torch.float16
        if model_id == "THUDM/CogVideoX-5b":
            if "variant" in kwargs:
                del kwargs["variant"]
            logger.info("TextToVideoPipeline loading bf16 variant for %s", model_id)
            kwargs["torch_dtype"] = torch.bfloat16

        self.model_id = model_id
        if model_id == "THUDM/CogVideoX-2b" or model_id == "THUDM/CogVideoX-5b":
            self.ldm = CogVideoXPipeline.from_pretrained(model_id, **kwargs)
        else:
            self.ldm = DiffusionPipeline.from_pretrained(model_id, **kwargs)
        self.ldm.vae.enable_slicing()
        self.ldm.vae.enable_tiling()
        self.ldm.to(get_torch_device())

        if os.environ.get("SFAST"):
            logger.info(
                "TextToVideoPipeline will be dynamicallly compiled with stable-fast for %s",
                model_id,
            )
            from app.pipelines.sfast import compile_model

            self.ldm = compile_model(self.ldm)

    def __call__(self, prompt: str, **kwargs) -> List[List[PIL.Image]]:
        if self.model_id == "THUDM/CogVideoX-2b" or self.model_id == "THUDM/CogVideoX-5b":
            if "motion_bucket_id" in kwargs:
                del kwargs["motion_bucket_id"]
            if "fps" in kwargs:
                del kwargs["fps"]
            if "noise_aug_strength" in kwargs:
                del kwargs["noise_aug_strength"]
            if "safety_check" in kwargs:
                del kwargs["safety_check"]
            if "width" in kwargs:
                del kwargs["width"]
            if "height" in kwargs:
                del kwargs["height"]
            kwargs["num_frames"] = 49
            kwargs["num_videos_per_prompt"] = 1

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
