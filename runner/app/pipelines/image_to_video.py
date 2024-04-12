from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir

from diffusers import StableVideoDiffusionPipeline, I2VGenXLPipeline
from huggingface_hub import file_download
import torch
import PIL
from typing import List
import logging
import os

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

I2VGEN_LIGHTNING_MODEL_ID = "ali-vilab/i2vgen-xl"
SVD_LIGHTNING_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"

class ImageToVideoPipeline(Pipeline):
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
            logger.info("ImageToVideoPipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        self.model_id = model_id

        if I2VGEN_LIGHTNING_MODEL_ID in model_id:
            self.ldm = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
        else:
            self.ldm = StableVideoDiffusionPipeline.from_pretrained(model_id, **kwargs)
        self.ldm.enable_vae_slicing()
        self.ldm.to(get_torch_device())

        if os.environ.get("SFAST"):
            logger.info(
                "ImageToVideoPipeline will be dynamicallly compiled with stable-fast for %s",
                model_id,
            )
            from app.pipelines.sfast import compile_model

            self.ldm = compile_model(self.ldm)

    def __call__(self, image: PIL.Image, **kwargs) -> List[List[PIL.Image]]:
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

        if SVD_LIGHTNING_MODEL_ID in self.model_id:
            if "decode_chunk_size" not in kwargs:
                kwargs["decode_chunk_size"] = 4
            if "prompt" in kwargs:
                del kwargs["prompt"]
        elif I2VGEN_LIGHTNING_MODEL_ID in self.model_id:
            kwargs["num_frames"] = 18
            kwargs["num_inference_steps"] = 50
            if "fps" in kwargs:
                del kwargs["fps"]
            if "motion_bucket_id" in kwargs:
                del kwargs["motion_bucket_id"]
            if "noise_aug_strength" in kwargs:
                del kwargs["noise_aug_strength"]
            prompt = ""
            if "prompt" in kwargs:
                prompt = kwargs["prompt"]
                del kwargs["prompt"]
            return self.ldm(prompt, image, **kwargs).frames

        return self.ldm(image, **kwargs).frames

    def __str__(self) -> str:
        return f"ImageToVideoPipeline model_id={self.model_id}"
