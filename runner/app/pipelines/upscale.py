import logging
import os
from compel import Compel, ReturnedEmbeddingsType 
from typing import List, Optional, Tuple

import PIL
import torch
from diffusers import StableDiffusionUpscalePipeline
from huggingface_hub import file_download
from PIL import ImageFile

from app.pipelines.base import Pipeline
from app.pipelines.utils import (
    SafetyChecker,
    get_model_dir,
    get_torch_device,
    is_lightning_model,
    is_turbo_model,
)
from app.utils.errors import InferenceError

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


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
                "UpscalingPiepline will NOT be optimized with DeepCache for %s",
                model_id,
            )

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

        # trying differnt configs of promp_embed for different models
        try:
            compel_proc=Compel(tokenizer=self.ldm.tokenizer, text_encoder=self.ldm.text_encoder)
            prompt_embeds = compel_proc(prompt)
            outputs = self.ldm(prompt_embeds=prompt_embeds, image=image, **kwargs)
        except Exception as e:
            logging.info(f"Failed to generate prompt embeddings: {e}. Using prompt and pooled embeddings.")

            try:
                compel_proc = Compel(tokenizer=[self.ldm.tokenizer, self.ldm.tokenizer_2],
                                text_encoder=[self.ldm.text_encoder, self.ldm.text_encoder_2],
                                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                                requires_pooled=[False, True])
                prompt_embeds, pooled_prompt_embeds = compel_proc(prompt)
                outputs = self.ldm(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    image=image,
                    **kwargs
                )
            except Exception as e:
                logging.info(f"Failed to generate prompt and pooled embeddings: {e}. Trying normal prompt.")
                try:
                    outputs = self.ldm(prompt, image=image, **kwargs)
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
