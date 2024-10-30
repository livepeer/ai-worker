# runner/app/pipelines/inpainting.py

import logging
import os
from typing import List, Optional, Tuple

import PIL
import torch
from app.pipelines.base import Pipeline
from app.pipelines.utils import (
    LoraLoader,
    SafetyChecker,
    get_model_dir,
    get_torch_device,
    split_prompt,
)
from app.utils.errors import InferenceError
from diffusers import AutoPipelineForInpainting, EulerAncestralDiscreteScheduler
from huggingface_hub import file_download
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

class InpaintingPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        
        # Load fp16 variant if available
        has_fp16_variant = any(
            ".fp16.safetensors" in fname
            for _, _, files in os.walk(folder_path)
            for fname in files
        )
        if torch_device != "cpu" and has_fp16_variant:
            logger.info("InpaintingPipeline loading fp16 variant for %s", model_id)
            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        self.ldm = AutoPipelineForInpainting.from_pretrained(
            model_id,
            safety_checker=None,  # We'll use our own safety checker
            **kwargs
        ).to(torch_device)

        # Enable memory efficient attention if available
        if hasattr(self.ldm, "enable_xformers_memory_efficient_attention"):
            logger.info("Enabling xformers memory efficient attention")
            self.ldm.enable_xformers_memory_efficient_attention()

        # Initialize safety checker on specified device
        safety_checker_device = os.getenv("SAFETY_CHECKER_DEVICE", "cuda").lower()
        self._safety_checker = SafetyChecker(device=safety_checker_device)
        
        # Initialize LoRA support
        self._lora_loader = LoraLoader(self.ldm)

    def __call__(
        self, prompt: str, image: PIL.Image, mask_image: PIL.Image, **kwargs
    ) -> Tuple[List[PIL.Image], List[Optional[bool]]]:
        # Extract parameters
        seed = kwargs.pop("seed", None)
        safety_check = kwargs.pop("safety_check", True)
        loras_json = kwargs.pop("loras", "")

        # Handle seed generation
        if seed is not None:
            if isinstance(seed, int):
                kwargs["generator"] = torch.Generator(get_torch_device()).manual_seed(seed)
            elif isinstance(seed, list):
                kwargs["generator"] = [
                    torch.Generator(get_torch_device()).manual_seed(s) for s in seed
                ]

        # Handle LoRA loading
        if not loras_json:
            self._lora_loader.disable_loras()
        else:
            self._lora_loader.load_loras(loras_json)

        # Clean up inference steps if invalid
        if "num_inference_steps" in kwargs and (
            kwargs["num_inference_steps"] is None or kwargs["num_inference_steps"] < 1
        ):
            del kwargs["num_inference_steps"]

        # Split prompts if multiple are provided
        prompts = split_prompt(prompt, max_splits=3)
        kwargs.update(prompts)
        neg_prompts = split_prompt(
            kwargs.pop("negative_prompt", ""),
            key_prefix="negative_prompt",
            max_splits=3,
        )
        kwargs.update(neg_prompts)

        try:
            outputs = self.ldm(
                prompt=prompt,
                image=image,
                mask_image=mask_image,
                **kwargs
            )
        except Exception as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            raise InferenceError(original_exception=e)

        if safety_check:
            _, has_nsfw_concept = self._safety_checker.check_nsfw_images(outputs.images)
        else:
            has_nsfw_concept = [None] * len(outputs.images)

        return outputs.images, has_nsfw_concept

    def __str__(self) -> str:
        return f"InpaintingPipeline model_id={self.model_id}"