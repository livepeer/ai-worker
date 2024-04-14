from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir

from diffusers import (
    AutoPipelineForImage2Image,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from safetensors.torch import load_file
from huggingface_hub import file_download, hf_hub_download
import torch
import PIL
import random
from typing import List
import logging
import os

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

SDXL_LIGHTNING_MODEL_ID = "ByteDance/SDXL-Lightning"
PIX2PIX_MODEL_ID = "timbrooks/instruct-pix2pix"


class ImageToImagePipeline(Pipeline):
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
            or SDXL_LIGHTNING_MODEL_ID in model_id
        )
        if torch_device != "cpu" and has_fp16_variant:
            logger.info("ImageToImagePipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        self.model_id = model_id

        # Special case SDXL-Lightning because the unet for SDXL needs to be swapped
        if SDXL_LIGHTNING_MODEL_ID in model_id:
            base = "stabilityai/stable-diffusion-xl-base-1.0"

            # ByteDance/SDXL-Lightning-2step
            if "2step" in model_id:
                unet_id = "sdxl_lightning_2step_unet"
            # ByteDance/SDXL-Lightning-4step
            elif "4step" in model_id:
                unet_id = "sdxl_lightning_4step_unet"
            # ByteDance/SDXL-Lightning-8step
            elif "8step" in model_id:
                unet_id = "sdxl_lightning_8step_unet"
            else:
                # Default to 2step
                unet_id = "sdxl_lightning_2step_unet"

            unet = UNet2DConditionModel.from_config(
                base, subfolder="unet", cache_dir=kwargs["cache_dir"]
            ).to(torch_device, kwargs["torch_dtype"])
            unet.load_state_dict(
                load_file(
                    hf_hub_download(
                        SDXL_LIGHTNING_MODEL_ID,
                        f"{unet_id}.safetensors",
                        cache_dir=kwargs["cache_dir"],
                    ),
                    device=str(torch_device),
                )
            )

            self.ldm = StableDiffusionXLPipeline.from_pretrained(
                base, unet=unet, **kwargs
            ).to(torch_device)

            self.ldm.scheduler = EulerDiscreteScheduler.from_config(
                self.ldm.scheduler.config, timestep_spacing="trailing"
            )
        elif PIX2PIX_MODEL_ID in model_id:
            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"
            self.ldm = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_id, **kwargs
            ).to(torch_device)
        else:
            self.ldm = AutoPipelineForImage2Image.from_pretrained(
                model_id, **kwargs
            ).to(torch_device)

        if os.environ.get("SFAST"):
            logger.info(
                "ImageToImagePipeline will be dynamicallly compiled with stable-fast for %s",
                model_id,
            )
            from app.pipelines.sfast import compile_model

            self.ldm = compile_model(self.ldm)

    def __call__(self, prompt: str, image: PIL.Image, **kwargs) -> List[PIL.Image]:
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
        elif SDXL_LIGHTNING_MODEL_ID in self.model_id:
            # SDXL-Lightning models should have guidance_scale = 0 and use
            # the correct number of inference steps for the unet checkpoint loaded
            kwargs["guidance_scale"] = 0.0

            if "2step" in self.model_id:
                kwargs["num_inference_steps"] = 2
            elif "4step" in self.model_id:
                kwargs["num_inference_steps"] = 4
            elif "8step" in self.model_id:
                kwargs["num_inference_steps"] = 8
            else:
                # Default to 2step
                kwargs["num_inference_steps"] = 2
        elif PIX2PIX_MODEL_ID in self.model_id:
            kwargs["guidance_scale"] = round(random.uniform(6.0, 9.0), ndigits=2)
            kwargs["image_guidance_scale"] = round(random.uniform(1.2, 1.8), ndigits=2)
            kwargs["num_inference_steps"] = 50

        return self.ldm(prompt, image=image, **kwargs).images

    def __str__(self) -> str:
        return f"ImageToImagePipeline model_id={self.model_id}"
