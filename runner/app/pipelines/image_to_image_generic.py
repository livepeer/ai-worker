import logging
import numpy as np
import os
from enum import Enum
from typing import List, Optional, Tuple

import PIL
import torch
from diffusers import (
    AutoPipelineForInpainting,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
)
from huggingface_hub import file_download
from PIL import Image, ImageOps

from app.pipelines.base import Pipeline
from app.pipelines.utils import (
    LoraLoader,
    get_model_dir,
    get_torch_device,
)
from app.utils.errors import InferenceError

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enumeration for task types."""

    INPAINTING = "inpainting"
    OUTPAINTING = "outpainting"
    SKETCH_TO_IMAGE = "sketch_to_image"

    @classmethod
    def list(cls):
        return [task.value for task in cls]


class ImageToImageGenericPipeline(Pipeline):
    def __init__(self, model_id: str, task: str):
        kwargs = {"cache_dir": get_model_dir(), "torch_dtype": torch.float16}
        torch_device = get_torch_device()

        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        # Load the fp16 variant if fp16 'safetensors' files are present in the cache.
        # NOTE: Exception for SDXL-Lightning model: despite having fp16 'safetensors'
        # files, they are not named according to the standard convention.
        has_fp16_variant = any(
            ".fp16.safetensors" in fname
            for _, _, files in os.walk(folder_path)
            for fname in files
        )
        if torch_device.type != "cpu" and has_fp16_variant:
            logger.info(
                "ImageToImageGenericPipeline loading fp16 variant for %s", model_id
            )

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        if task not in TaskType.list():
            raise ValueError(f"Unsupported task: {task}")

        self.task = task

        # Initialize pipelines based on task
        if self.task == TaskType.INPAINTING.value:
            self.pipeline = AutoPipelineForInpainting.from_pretrained(
                model_id, **kwargs
            ).to(torch_device)
            self.pipeline.enable_model_cpu_offload()

        elif self.task == TaskType.OUTPAINTING.value:
            self.controlnet = (
                ControlNetModel.from_pretrained(
                    model_id, torch_dtype=torch.float16, variant="fp16"
                ).to(torch_device),
            )
            self.vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
            ).to(torch_device)
            self.pipeline_stage1 = StableDiffusionXLControlNetPipeline.from_pretrained(
                "SG161222/RealVisXL_V4.0",
                controlnet=self.controlnet,
                vae=self.vae,
                safety_checker=None,
                **kwargs,
            ).to(torch_device)
            self.pipeline_stage2 = StableDiffusionXLInpaintPipeline.from_pretrained(
                "OzzyGT/RealVisXL_V4.0_inpainting", vae=self.vae, **kwargs
            ).to(torch_device)

        elif self.task == TaskType.SKETCH_TO_IMAGE.value:
            self.controlnet = ControlNetModel.from_pretrained(model_id, **kwargs).to(
                torch_device
            )
            self.vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", **kwargs
            ).to(torch_device)
            eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
            )
            self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=self.controlnet,
                vae=self.vae,
                safety_checker=None,
                scheduler=eulera_scheduler,
                **kwargs,
            ).to(torch_device)

        self._lora_loader = LoraLoader(self.pipeline)

        if self.task == TaskType.OUTPAINTING.value:
            self._lora_loader1 = LoraLoader(self.pipeline_stage1)
            self._lora_loader2 = LoraLoader(self.pipeline_stage2)

    def __call__(
        self,
        prompt: List[str],
        image: PIL.Image.Image,
        mask_image: Optional[PIL.Image.Image] = None,
        **kwargs,
    ) -> Tuple[List[PIL.Image], List[Optional[bool]]]:
        # Handle num_inference_steps and other model-specific settings
        if "num_inference_steps" in kwargs and (
            kwargs["num_inference_steps"] is None or kwargs["num_inference_steps"] < 1
        ):
            del kwargs["num_inference_steps"]

        # Extract parameters from kwargs
        seed = kwargs.pop("seed", None)
        safety_check = kwargs.pop("safety_check", True)
        loras_json = kwargs.pop("loras", "")
        guidance_scale = kwargs.pop("guidance_scale", None)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        controlnet_conditioning_scale = kwargs.pop(
            "controlnet_conditioning_scale", None
        )
        control_guidance_end = kwargs.pop("control_guidance_end", None)
        strength = kwargs.pop("strength", None)

        if len(prompt) == 1:
            prompt = prompt[0]

        # Handle seed initialization for reproducibility
        if seed is not None:
            if isinstance(seed, int):
                kwargs["generator"] = torch.Generator(get_torch_device()).manual_seed(
                    seed
                )
            elif isinstance(seed, list):
                kwargs["generator"] = [
                    torch.Generator(get_torch_device()).manual_seed(s) for s in seed
                ]

        # Dynamically (un)load LoRas.
        if not loras_json:
            if self.task == TaskType.OUTPAINTING.value:
                self._lora_loader1.disable_loras()
                self._lora_loader2.disable_loras()
            else:
                self._lora_loader.disable_loras()  # Assuming _lora_loader is defined elsewhere
        else:
            if self.task == TaskType.OUTPAINTING.value:
                self._lora_loader1.load_loras(loras_json)
                self._lora_loader2.load_loras(loras_json)
            else:
                self._lora_loader.load_loras(
                    loras_json
                )  # Assuming _lora_loader is defined elsewhere

        # Handle num_inference_steps and other model-specific settings
        if "num_inference_steps" in kwargs and (
            kwargs["num_inference_steps"] is None or kwargs["num_inference_steps"] < 1
        ):
            del kwargs["num_inference_steps"]

        # Ensure proper inference configuration based on model
        if self.task == TaskType.INPAINTING.value:
            if mask_image is None:
                raise ValueError("Mask image is required for inpainting.")
            try:
                outputs = self.pipeline(
                    prompt=prompt,
                    image=image,
                    mask_image=mask_image,
                    guidance_scale=guidance_scale[0],
                    strength=strength,
                    **kwargs,
                ).images[0]
            except torch.cuda.OutOfMemoryError as e:
                raise e
            except Exception as e:
                raise InferenceError(original_exception=e)
        elif self.task == TaskType.OUTPAINTING.value:
            try:
                resized_image, white_bg_image = self._scale_and_paste(image)
                temp_image = self.pipeline_stage1(
                    prompt=prompt[0],
                    image=white_bg_image,
                    guidance_scale=guidance_scale[0],
                    num_inference_steps=num_inference_steps[0],
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    control_guidance_end=control_guidance_end,
                    **kwargs,
                ).images[0]

                x = (1024 - resized_image.width) // 2
                y = (1024 - resized_image.height) // 2
                temp_image.paste(resized_image, (x, y), resized_image)

                mask = Image.new("L", temp_image.size)
                mask.paste(resized_image.split()[3], (x, y))
                mask = ImageOps.invert(mask)
                final_mask = mask.point(lambda p: p > 128 and 255)
                mask_blurred = self.pipeline_stage2.mask_processor.blur(
                    final_mask, blur_factor=20
                )

                outputs = self.pipeline_stage2(
                    prompt[1],
                    image=temp_image,
                    mask_image=mask_blurred,
                    strength=strength,
                    guidance_scale=guidance_scale[1],
                    num_inference_steps=num_inference_steps[1],
                    **kwargs,
                ).images[0]

                x = (1024 - resized_image.width) // 2
                y = (1024 - resized_image.height) // 2
                outputs.paste(resized_image, (x, y), resized_image)
            except torch.cuda.OutOfMemoryError as e:
                raise e
            except Exception as e:
                raise InferenceError(original_exception=e)
        elif self.task == TaskType.SKETCH_TO_IMAGE.value:
            try:
                # must resize to 1024*1024 or same resolution bucket to get the best performance
                width, height = image.size
                ratio = np.sqrt(1024.0 * 1024.0 / (width * height))
                new_width, new_height = int(width * ratio), int(height * ratio)
                image = image.resize((new_width, new_height))
                outputs = self.pipeline(
                    prompt=prompt,
                    image=image,
                    num_inference_steps=num_inference_steps[0],
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    **kwargs,
                ).images[0]
            except torch.cuda.OutOfMemoryError as e:
                raise e
            except Exception as e:
                raise InferenceError(original_exception=e)

        # Safety check for NSFW content
        if safety_check:
            _, has_nsfw_concept = self._safety_checker.check_nsfw_images(outputs.images)
        else:
            has_nsfw_concept = [None] * len(outputs.images)

        return outputs, has_nsfw_concept  # Return the first image in the output list

    @staticmethod
    def _scale_and_paste(
        original_image: PIL.Image.Image,
    ) -> Tuple[PIL.Image.Image, PIL.Image.Image]:
        """Resize and paste the original image onto a 1024x1024 white canvas."""
        aspect_ratio = original_image.width / original_image.height
        if original_image.width > original_image.height:
            new_width = 1024
            new_height = round(new_width / aspect_ratio)
        else:
            new_height = 1024
            new_width = round(new_height * aspect_ratio)

        resized_original = original_image.resize((new_width, new_height), Image.LANCZOS)
        white_background = Image.new("RGBA", (1024, 1024), "white")
        x = (1024 - new_width) // 2
        y = (1024 - new_height) // 2
        white_background.paste(resized_original, (x, y), resized_original)

        return resized_original, white_background

    def __str__(self) -> str:
        return f"ImageToImageGenericPipeline task={self.task}"
