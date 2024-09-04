from app.pipelines.base import Pipeline

from app.pipelines.utils import (
    SafetyChecker,
    get_model_dir,
    get_torch_device,
)

from diffusers import CogVideoXPipeline, DiffusionPipeline
from huggingface_hub import file_download
import torch
import PIL
from typing import List
import logging
import os
from diffusers.utils import BaseOutput
import inspect
import time

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
            self.ldm = None
            self.ldm2 = None
            if os.environ.get("COGVIDEOX_DEVICE_MAP_2_GPU", "") != "":
                #setup transformer for GPU 0
                self.ldm = CogVideoXPipeline.from_pretrained(model_id, tokenizer=None, text_encoder=None, vae=None, **kwargs).to("cuda:0")
                #setup all other components for GPU 1
                self.ldm2 = CogVideoXPipeline.from_pretrained(model_id, transformer=None, **kwargs).to("cuda:1")
                self.ldm2.vae.enable_tiling()
            else:   
                self.ldm = DiffusionPipeline.from_pretrained(model_id, **kwargs)
                self.ldm.enable_model_cpu_offload()
                self.ldm.vae.enable_tiling()

        if os.environ.get("SFAST"):
            logger.info(
                "TextToVideoPipeline will be dynamicallly compiled with stable-fast for %s",
                model_id,
            )
            from app.pipelines.sfast import compile_model

            self.ldm = compile_model(self.ldm)
        
        safety_checker_device = os.getenv("SAFETY_CHECKER_DEVICE", "cuda").lower()
        self._safety_checker = SafetyChecker(device=safety_checker_device)

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
        
        safety_check = kwargs.pop("safety_check", True)

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

        output = None
        if os.environ.get("COGVIDEOX_DEVICE_MAP_2_GPU", "") != "":
            with torch.no_grad():
                #generate prompt embeds on GPU 1
                start = time.time()
                prompt_embeds = negative_prompt_embeds = None
                encode_kwargs = inspect.signature(self.ldm2.encode_prompt).parameters.keys()
                negative_prompt = kwargs.pop("negative_prompt", "")
                prompt_embeds, negative_prompt_embeds = self.ldm2.encode_prompt(prompt, negative_prompt, **{k: v for k, v in kwargs.items() if k in encode_kwargs})
                logger.info(f"encode_prompt took: {time.time()-start} seconds")
                #generate video on GPU 0
                start = time.time()
                prompt_embeds = prompt_embeds.to(self.ldm._execution_device)
                negative_prompt_embeds = negative_prompt_embeds.to(self.ldm._execution_device)
                logger.info(f"prompt embeds conversion took: {time.time()-start} seconds")
                start = time.time()
                ldm_kwargs = inspect.signature(self.ldm.__call__).parameters.keys()
                latents = self.ldm(prompt=None, negative_prompt=None,
                                   prompt_embeds=prompt_embeds, 
                                   negative_prompt_embeds=negative_prompt_embeds, 
                                   output_type="latent", return_dict=False,
                                   **{k: v for k, v in kwargs.items() if k in ldm_kwargs})
                logger.info(f"transformer took: {time.time()-start} seconds")
                #use the VAE on GPU 1 to process to image
                #copied from diffusers/pipelines/cogvideo/pipeline_cogvideox L719
                start = time.time()
                latents = latents[0].to(self.ldm2._execution_device)
                logger.info(f"latents conversion took: {time.time()-start} seconds")
                start = time.time()
                video = self.ldm2.decode_latents(latents)
                video = self.ldm2.video_processor.postprocess_video(video=video, output_type="pil") #only support default output_type="pil"
                logger.info(f"vae decode took: {time.time()-start} seconds")

                output = BaseOutput(frames=video)

        else:
            output = self.ldm(prompt, **kwargs)
        
        if safety_check:
            #checks first frame, last frame and middle frame
            start = time.time()
            _, has_nsfw_concept = self._safety_checker.check_nsfw_images(output.frames[0])
            if not has_nsfw_concept:
                _, has_nsfw_concept = self._safety_checker.check_nsfw_images(output.frames[-1])
            if not has_nsfw_concept:
                _, has_nsfw_concept = self._safety_checker.check_nsfw_images(output.frames[len(output.frames)//2])
            logger.info(f"safety checker took: {time.time()-start} seconds")
        else:
            has_nsfw_concept = [None]

        return output.frames, has_nsfw_concept

    def __str__(self) -> str:
        return f"TextToVideoPipeline model_id={self.model_id}"
