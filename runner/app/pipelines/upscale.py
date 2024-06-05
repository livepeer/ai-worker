from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir, SafetyChecker

from diffusers import (
    StableDiffusionUpscalePipeline
)
from safetensors.torch import load_file
from huggingface_hub import file_download, hf_hub_download
import torch
import PIL
from typing import List, Tuple, Optional
import logging
import os

from PIL import ImageFile
from PIL import Image
from io import BytesIO
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

SDX4_UPSCALER_MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"


class UpscalePipeline(Pipeline):
    def __init__(self, model_id: str):
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        has_fp16_variant = (
                any(
                    ".fp16.safetensors" in fname
                    for _, _, files in os.walk(folder_path)
                    for fname in files
                )
                or SDX4_UPSCALER_MODEL_ID in model_id
        )
        if torch_device != "cpu" and has_fp16_variant:
            logger.info("UpscalePipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        self.model_id = model_id
        self.ldm = StableDiffusionUpscalePipeline.from_pretrained(
                model_id, **kwargs
            ).to(torch_device)

        sfast_enabled = os.getenv("SFAST", "").strip().lower() == "true"
        deepcache_enabled = os.getenv("DEEPCACHE", "").strip().lower() == "true"
        if sfast_enabled and deepcache_enabled:
            logger.warning(
                "Both 'SFAST' and 'DEEPCACHE' are enabled. This is not recommended "
                "as it may lead to suboptimal performance. Please disable one of them."
            )

        if sfast_enabled:
            logger.info(
                "UpscalePipeline will be dynamically compiled with stable-fast "
                "for %s",
                model_id,
            )
            from app.pipelines.optim.sfast import compile_model

            self.ldm = compile_model(self.ldm)

            # Warm-up the pipeline.
            # TODO: Not yet supported for UpscalePipeline.
            if os.getenv("SFAST_WARMUP", "true").lower() == "true":
                logger.warning(
                    "The 'SFAST_WARMUP' flag is not yet supported for the "
                    "UpscalePipeline and will be ignored. As a result the first "
                    "call may be slow if 'SFAST' is enabled."
                )

        if deepcache_enabled:
            logger.info(
                "UpscalePipeline will be optimized with DeepCache for %s",
                model_id,
            )
            from app.pipelines.optim.deepcache import enable_deepcache

            self.ldm = enable_deepcache(self.ldm)

        safety_checker_device = os.getenv("SAFETY_CHECKER_DEVICE", "cuda").lower()
        self._safety_checker = SafetyChecker(device=safety_checker_device)

    def __call__(
            self, prompt: str, image: PIL.Image, **kwargs
    ) -> Tuple[List[PIL.Image], List[Optional[bool]]]:
        seed = kwargs.pop("seed", None)
        safety_check = kwargs.pop("safety_check", True)

        if seed is not None:
            if isinstance(seed, int):
                kwargs["generator"] = torch.Generator(get_torch_device()).manual_seed(
                    seed
                )
            elif isinstance(seed, list):
                kwargs["generator"] = [
                    torch.Generator(get_torch_device()).manual_seed(s) for s in seed
                ]

        output = self.ldm(prompt, image=image, **kwargs)

        if safety_check:
            _, has_nsfw_concept = self._safety_checker.check_nsfw_images(output.images)
        else:
            has_nsfw_concept = [None] * len(output.images)

        return output.images, has_nsfw_concept

    def __str__(self) -> str:
        return f"UpscalePipeline model_id={self.model_id}"
