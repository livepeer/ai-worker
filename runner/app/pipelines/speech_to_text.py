from fastapi import File
from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from huggingface_hub import file_download
import torch
import PIL
from typing import List
import logging
import os

logger = logging.getLogger(__name__)


class SpeechToTextPipeline(Pipeline):
    def __init__(self, model_id: str):
        # kwargs = {"cache_dir": get_model_dir()}
        kwargs = {}

        torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        # Load fp16 variant if fp16 safetensors files are found in cache
        # Special case SDXL-Lightning because the safetensors files are fp16 but are not
        # named properly right now
        has_fp16_variant = any(
            ".fp16.safetensors" in fname
            for _, _, files in os.walk(folder_path)
            for fname in files
        )
        if torch_device != "cpu" and has_fp16_variant:
            logger.info("SpeechToTextPipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        if os.environ.get("BFLOAT16"):
            logger.info("SpeechToTextPipeline using bfloat16 precision for %s", model_id)
            kwargs["torch_dtype"] = torch.bfloat16

        self.model_id = model_id

        import time
        start_time = time.time()
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, low_cpu_mem_usage=True, use_safetensors=True, **kwargs
        ).to(torch_device)

        processor = AutoProcessor.from_pretrained(model_id)
        print(f"Time taken to load model: {time.time() - start_time:.2f}s")

        self.ldm = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            **kwargs,
        )

    def __call__(self, audio: str, **kwargs) -> List[File]:
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

        result = self.ldm(audio, **kwargs)
        return result

    def __str__(self) -> str:
        return f"SpeechToTextPipeline model_id={self.model_id}"
