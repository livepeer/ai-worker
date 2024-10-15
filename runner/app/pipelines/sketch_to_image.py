import logging
import os
from enum import Enum
from typing import List, Optional, Tuple

import PIL
import torch
from app.pipelines.base import Pipeline
from app.pipelines.utils import (
    LoraLoader,
    SafetyChecker,
    get_model_dir,
    get_torch_device,
    is_lightning_model,
    is_turbo_model,
)
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetPipeline,
)
from huggingface_hub import file_download, hf_hub_download
from PIL import ImageFile
from safetensors.torch import load_file

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class ModelName(Enum):
    """Enumeration mapping model names to their corresponding IDs."""

    SCRIBBLE_SDXL = "xinsir/controlnet-scribble-sdxl-1.0"

    @classmethod
    def list(cls):
        """Return a list of all model IDs."""
        return list(map(lambda c: c.value, cls))


class SketchToImagePipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
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
        )

        torch_dtype = torch.float
        if torch_device.type != "cpu" and has_fp16_variant:
            logger.info("SketchToImagePipeline loading fp16 variant for %s", model_id)
            torch_dtype = torch.float16
            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            subfolder="scheduler",
            cache_dir=get_model_dir(),
        )
        controlnet = ControlNetModel.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            cache_dir=get_model_dir(),
        )
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch_dtype
        )
        self.ldm = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            safety_checker=None,
            scheduler=eulera_scheduler,
        ).to(torch_device)

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
        if "num_inference_steps" in kwargs and (
            kwargs["num_inference_steps"] is None or kwargs["num_inference_steps"] < 1
        ):
            del kwargs["num_inference_steps"]

        output = self.ldm(prompt, image=image, **kwargs)

        if safety_check:
            _, has_nsfw_concept = self._safety_checker.check_nsfw_images(output.images)
        else:
            has_nsfw_concept = [None] * len(output.images)

        return output.images, has_nsfw_concept

    def __str__(self) -> str:
        return f"SketchToImagePipeline model_id={self.model_id}"
