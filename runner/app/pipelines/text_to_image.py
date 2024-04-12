from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir

from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline
)
from safetensors.torch import load_file
from huggingface_hub import file_download, hf_hub_download
import torch
import PIL
from typing import List
import logging
import os

logger = logging.getLogger(__name__)

SDXL_LIGHTNING_MODEL_ID = "ByteDance/SDXL-Lightning"
SDXL_BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


class TextToImagePipeline(Pipeline):
    def __init__(self, model_id: str):
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        # Load fp16 variant if fp16 safetensors files are found in cache
        # Special case SDXL-Lightning because the safetensors files are fp16 but are not
        # named properly right now
        has_fp16_variant = (
            any(
                ".fp16.safetensors" in fname
                for _, _, files in os.walk(folder_path)
                for fname in files
            )
            or SDXL_LIGHTNING_MODEL_ID in model_id
        )
        if torch_device != "cpu" and has_fp16_variant:
            logger.info("TextToImagePipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        if os.environ.get("BFLOAT16"):
            logger.info("TextToImagePipeline using bfloat16 precision for %s", model_id)
            kwargs["torch_dtype"] = torch.bfloat16

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
        elif SDXL_BASE_MODEL_ID in self.model_id:
            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"
            kwargs["use_safetensors"] = True
            self.ldm = StableDiffusionXLPipeline.from_pretrained(model_id, **kwargs).to("cuda")
            self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=self.ldm.text_encoder_2,
                vae=self.ldm.vae,
                torch_dtype=kwargs["torch_dtype"],
                use_safetensors=True,
                variant=kwargs["variant"],
            ).to("cuda")

        else:
            self.ldm = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs).to(
                torch_device
            )

        if os.environ.get("TORCH_COMPILE"):
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True

            self.ldm.unet.to(memory_format=torch.channels_last)
            self.ldm.vae.to(memory_format=torch.channels_last)

            self.ldm.unet = torch.compile(
                self.ldm.unet, mode="max-autotune", fullgraph=True
            )
            self.ldm.vae.decode = torch.compile(
                self.ldm.vae.decode, mode="max-autotune", fullgraph=True
            )

        if os.environ.get("SFAST"):
            logger.info(
                "TextToImagePipeline will be dynamicallly compiled with stable-fast for %s",
                model_id,
            )
            from app.pipelines.sfast import compile_model

            self.ldm = compile_model(self.ldm)

    def __call__(self, prompt: str, **kwargs) -> List[PIL.Image]:
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

            if "num_inference_steps" not in kwargs:
                kwargs["num_inference_steps"] = 1
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
        elif SDXL_BASE_MODEL_ID in self.model_id:
            kwargs["num_inference_steps"] = 40
            kwargs["denoising_end"] = 0.8
            kwargs["output_type"] = "latent"
            image = self.ldm(prompt, **kwargs).images
            del kwargs["output_type"]
            del kwargs["denoising_end"]
            kwargs["image"] = image
            kwargs["denoising_start"] = 0.8
            return self.refiner(prompt, **kwargs).images

        return self.ldm(prompt, **kwargs).images

    def __str__(self) -> str:
        return f"TextToImagePipeline model_id={self.model_id}"
