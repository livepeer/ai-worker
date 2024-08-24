import logging
import os
from typing import List, Optional, Tuple
from enum import Enum

import PIL
import PIL.Image
import torch
from diffusers import StableDiffusionUpscalePipeline
from huggingface_hub import file_download, hf_hub_download
from PIL import ImageFile, Image
from spandrel import ModelLoader
import numpy as np

from app.pipelines.base import Pipeline
from app.pipelines.utils import (
    SafetyChecker,
    get_model_dir,
    get_torch_device,
    is_lightning_model,
    is_turbo_model
)
from app.utils.errors import InferenceError

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

class ModelName(Enum):
    """Enumeration mapping model names to their corresponding IDs."""

    BSRGAN_X2 = "bsrgan_x2"
    BSRGAN_X4 = "bsrgan_x4"
    NMKD_SUPERSCALE_SP_X4 = "nmkd_superscale_sp_x4"
    NMKD_TYPESCALE_X8 = "nmkd_typescale_x4"
    REALESRGAN_X2 = "realesrgan_x2"
    REALESRGAN_X4 = "realesrgan_x4"
    REALESRGAN_ANIME_X4 = "realesrgan_anime_x4"
    REAL_HAT_GAN_SR_X4 = "real_hat_gan_sr_x4"
    REAL_HAT_GAN_SR_X4_SHARPER = "real_hat_gan_sr_x4_sharper"
    SCUNET_COLOR_GAN = "scunet_color_real_gan"
    SCUNET_COLOR_PSNR = "scunet_color_real_psnr"
    SWIN2SR_CLASSICAL_X2 = "swin2sr_classical_x2"
    SWIN2SR_CLASSICAL_X4 = "swin2sr_classical_x4"
    

    @classmethod
    def list(cls):
        """Return a list of all model IDs."""
        return list(map(lambda c: c.value, cls))
    @classmethod
    def get_model_file(cls, model):
        match model:
            case cls.BSRGAN_X2.value:
                return hf_hub_download("ad-astra-video/upscalers", filename="BSRGANx2.pth", cache_dir=get_model_dir())
            case cls.BSRGAN_X4.value:
                return hf_hub_download("ad-astra-video/upscalers", filename="BSRGANx4.pth", cache_dir=get_model_dir())
            case cls.NMKD_SUPERSCALE_SP_X4.value:
                return hf_hub_download("ad-astra-video/upscalers", filename="4x_NMKD-Superscale-SP_178000_G.pth", cache_dir=get_model_dir())
            case cls.NMKD_TYPESCALE_X8.value:
                return hf_hub_download("ad-astra-video/upscalers", filename="8x_NMKD-Typescale_175k.pth", cache_dir=get_model_dir())
            case cls.REALESRGAN_X2.value:
                return hf_hub_download("ad-astra-video/upscalers", filename="RealESRGAN_x2plus.pth", cache_dir=get_model_dir())
            case cls.REALESRGAN_X4.value:
                return hf_hub_download("ad-astra-video/upscalers", filename="RealESRGAN_x4plus.pth", cache_dir=get_model_dir())
            case cls.REALESRGAN_ANIME_X4.value:
                return hf_hub_download("ad-astra-video/upscalers", filename="RealESRGAN_x4plus_anime_6B.pth", cache_dir=get_model_dir())
            case cls.REAL_HAT_GAN_SR_X4.value:
                return hf_hub_download("ad-astra-video/upscalers", filename="Real_HAT_GAN_SRx4.pth", cache_dir=get_model_dir())
            case cls.REAL_HAT_GAN_SR_X4_SHARPER.value:
                return hf_hub_download("ad-astra-video/upscalers", filename="Real_HAT_GAN_sharper.pth", cache_dir=get_model_dir())
            case cls.SCUNET_COLOR_GAN.value:
                return hf_hub_download("ad-astra-video/upscalers", filename="scunet_color_real_gan.pth", cache_dir=get_model_dir())
            case cls.SCUNET_COLOR_PSNR.value:
                return hf_hub_download("ad-astra-video/upscalers", filename="scunet_color_real_psnr.pth", cache_dir=get_model_dir())
            case cls.SWIN2SR_CLASSICAL_X2.value:
                return hf_hub_download("ad-astra-video/upscalers", filename="Swin2SR_ClassicalSR_X2_64.pth", cache_dir=get_model_dir())
            case cls.SWIN2SR_CLASSICAL_X4.value:
                return hf_hub_download("ad-astra-video/upscalers", filename="Swin2SR_ClassicalSR_X4_64.pth", cache_dir=get_model_dir())
            
    @classmethod
    def get_model_scale(cls, model):
        match model:
            case cls.BSRGAN_X2.value:
                return 2
            case cls.BSRGAN_X4.value:
                return 4
            case cls.NMKD_SUPERSCALE_SP_X4.value:
                return 4
            case cls.NMKD_TYPESCALE_X8.value:
                return 8
            case cls.REALESRGAN_X2.value:
                return 2
            case cls.REALESRGAN_X4.value:
                return 4
            case cls.REALESRGAN_ANIME_X4.value:
                return 4
            case cls.REAL_HAT_GAN_SR_X4.value:
                return 4
            case cls.REAL_HAT_GAN_SR_X4_SHARPER.value:
                return 4
            case cls.SCUNET_COLOR_GAN.value:
                return 4
            case cls.SCUNET_COLOR_PSNR.value:
                return 4
            case cls.SWIN2SR_CLASSICAL_X2.value:
                return 2
            case cls.SWIN2SR_CLASSICAL_X4.value:
                return 4
            
        
class UpscalePipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
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
        if torch_device.type != "cpu" and has_fp16_variant:
            logger.info("UpscalePipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        if self.model_id == "stabilityai/stable-diffusion-x4-upscaler":
            self.ldm = StableDiffusionUpscalePipeline.from_pretrained(
                model_id, **kwargs
            ).to(torch_device)
        else:
            if self.model_id in ModelName.list():
                #use spandrel to load the model
                model_file = ModelName.get_model_file(self.model_id)
                logger.info(f"loading model file: {model_file}")
                model = ModelLoader().load_from_file(model_file)
                logger.info(f"model loaded, scale={model.scale}")
                # send it to the GPU and put it in inference mode
                model.cuda().eval()
                self.ldm = model
            else:
                raise ValueError("Model not supported")

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

        if deepcache_enabled and not (
            is_lightning_model(model_id) or is_turbo_model(model_id)
        ):
            logger.info(
                "UpscalePipeline will be optimized with DeepCache for %s",
                model_id,
            )
            from app.pipelines.optim.deepcache import enable_deepcache

            self.ldm = enable_deepcache(self.ldm)
        elif deepcache_enabled:
            logger.warning(
                "DeepCache is not supported for Lightning or Turbo models. "
                "TextToImagePipeline will NOT be optimized with DeepCache for %s",
                model_id,
            )

        safety_checker_device = os.getenv("SAFETY_CHECKER_DEVICE", "cuda").lower()
        self._safety_checker = SafetyChecker(device=safety_checker_device)

    def __call__(
        self, prompt: str, image: PIL.Image, **kwargs
    ) -> Tuple[List[PIL.Image], List[Optional[bool]]]:
        seed = kwargs.pop("seed", None)
        safety_check = kwargs.pop("safety_check", True)
        torch_device = get_torch_device()
        if seed is not None:
            if isinstance(seed, int):
                kwargs["generator"] = torch.Generator(torch_device).manual_seed(
                    seed
                )
            elif isinstance(seed, list):
                kwargs["generator"] = [
                    torch.Generator(torch_device).manual_seed(s) for s in seed
                ]

        if "num_inference_steps" in kwargs and (
            kwargs["num_inference_steps"] is None or kwargs["num_inference_steps"] < 1
        ):
            del kwargs["num_inference_steps"]

        try:
            if self.model_id == "stabilityai/stable-diffusion-x4-upscaler":   
                outputs = self.ldm(prompt, image=image, **kwargs)
            else:
                max_scale = self.get_max_scale_for_input(image)
                if self.ldm.scale > max_scale:
                    raise ValueError("requested scale too high")
                
                # Convert PIL image to NumPy array
                img_tensor = self.pil_to_tensor(image)
                img_tensor = img_tensor.to(torch_device)
                outputs = self.ldm(img_tensor)
                outputs = self.tensor_to_pil(outputs)
                outputs.images = [outputs]
        except torch.cuda.OutOfMemoryError as e:
            raise e
        except Exception as e:
            raise InferenceError(original_exception=e)

        if safety_check:
            _, has_nsfw_concept = self._safety_checker.check_nsfw_images(outputs.images)
        else:
            has_nsfw_concept = [None] * len(outputs.images)

        return outputs.images, has_nsfw_concept

    def __str__(self) -> str:
        return f"UpscalePipeline model_id={self.model_id}"
    
    # Load and preprocess the image
    def pil_to_tensor(self, image: PIL.Image) -> torch.Tensor:
        # Convert PIL image to NumPy array
        img = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor and reorder dimensions to (C, H, W)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0)
        
        return img

    # Postprocess the output from ESRGAN
    def tensor_to_pil(self, tensor: torch.Tensor) -> PIL.Image:
        # Remove the batch dimension and reorder dimensions to (H, W, C)
        img = tensor.squeeze(0).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        
        # Convert the pixel values to the [0, 255] range and convert to uint8
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        
        return Image.fromarray(img)
        
    def get_max_scale_for_input(self, image: PIL.Image) -> int:
        w, h = image.size        
        if (w*h) > 1048576: #1024x1024
            return 2
        elif (w*h) > 65536: #256x256
            return 4
        else:
            return 8




