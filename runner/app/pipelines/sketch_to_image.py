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

        eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            subfolder="scheduler"
        )
        controlnet = ControlNetModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16
        )
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16
        )
        self.ldm = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            safety_checker=None,
            torch_dtype=torch.float16,
            scheduler=eulera_scheduler,
        ).to(torch_device)

    def __call__(
        self, prompt: str, image: PIL.Image, **kwargs
    ) -> List[PIL.Image]:
        output = self.ldm(prompt, image=image, **kwargs)

        return output.images

    def __str__(self) -> str:
        return f"SketchToImagePipeline model_id={self.model_id}"
