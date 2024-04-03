from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir

from diffusers import StableVideoDiffusionPipeline
from huggingface_hub import file_download
import torch
import PIL
from typing import List
import logging
import os
import time

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

WARMUP_ITERATIONS = 3  # Warm-up calls count when SFAST is enabled.
WARMUP_BATCH_SIZE = 3  # Max batch size for warm-up calls when SFAST is enabled.


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
        self.ldm = StableVideoDiffusionPipeline.from_pretrained(model_id, **kwargs)
        self.ldm.to(get_torch_device())

        if os.environ.get("SFAST"):
            logger.info(
                "ImageToVideoPipeline will be dynamicallly compiled with stable-fast for %s",
                model_id,
            )
            from app.pipelines.sfast import compile_model

            self.ldm = compile_model(self.ldm)

            # Retrieve default model params.
            warmup_kwargs = {
                "image": PIL.Image.new("RGB", (512, 512)),
                "height": 512,
                "width": 512,
            }

            # NOTE: Warmup pipeline.
            # The initial calls will trigger compilation and might be very slow.
            # After that, it should be very fast.
            # FIXME: This will crash the pipeline if there is not enough VRAM available.
            logger.info("Warming up pipeline...")
            import time
            for ii in range(WARMUP_ITERATIONS):
                logger.info(f"Warmup iteration {ii + 1}...")
                t = time.time()
                try:
                    self.ldm(**warmup_kwargs).frames
                except Exception as e:
                    logger.error(f"ImageToVideoPipeline warmup error: {e}")
                    logger.exception(e)
                    # FIXME: When cuda out of memory, we need to reload the full model before it works again :(. torch.cuda.clear_cache() does not work.
                    # continue
                    raise e
                logger.info("Warmup iteration took %s seconds", time.time() - t)

    def __call__(self, image: PIL.Image, **kwargs) -> List[List[PIL.Image]]:
        if "decode_chunk_size" not in kwargs:
            kwargs["decode_chunk_size"] = 4

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

        t = time.time()
        frames = self.ldm(image, **kwargs).frames
        logger.info("TextToImagePipeline took %s seconds", time.time() - t)

        return frames
        # return self.ldm(image, **kwargs).frames

    def __str__(self) -> str:
        return f"ImageToVideoPipeline model_id={self.model_id}"
