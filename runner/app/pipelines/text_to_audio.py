mport logging
import os
from typing import List, Tuple
import io

import torch
import soundfile as sf
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from fastapi import UploadFile
from huggingface_hub import file_download
from diffusers import StableAudioPipeline

logger = logging.getLogger(__name__)

class TextToAudioPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {}

        torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        # Load fp16 variant if fp16 safetensors files are found in cache
        has_fp16_variant = any(
            ".fp16.safetensors" in fname
            for _, _, files in os.walk(folder_path)
            for fname in files
        )
        if torch_device != "cpu" and has_fp16_variant:
            logger.info("TextToAudioPipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        if os.environ.get("BFLOAT16"):
            logger.info("TextToAudioPipeline using bfloat16 precision for %s", model_id)
            kwargs["torch_dtype"] = torch.bfloat16

        logger.info(f"Loading StableAudioPipeline with model: {model_id}")
        self.tam = StableAudioPipeline.from_pretrained(
            model_id,
            cache_dir=get_model_dir(),
            **kwargs
        ).to(torch_device)

    def __call__(self, prompt: str, **kwargs) -> Tuple[List[io.BytesIO], List[bool]]:
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        audio_length_in_s = kwargs.get("audio_length_in_s", 10.0)
        negative_prompt = kwargs.get("negative_prompt", None)
        seed = kwargs.get("seed", None)

        if seed is not None:
            generator = torch.Generator(get_torch_device()).manual_seed(seed)
            kwargs["generator"] = generator

        audio = self.tam(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            audio_length_in_s=audio_length_in_s,
            **kwargs
        ).audios

        audio_buffers = []
        for output in audio:
            output = output.cpu().numpy()
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, output, self.tam.sampling_rate, format='wav')
            audio_buffer.seek(0)
            audio_buffers.append(audio_buffer)

        # StableAudioPipeline doesn't have a built-in safety checker
        has_nsfw_concept = [False] * len(audio_buffers)

        return audio_buffers, has_nsfw_concept

    def __str__(self) -> str:
        return f"TextToAudioPipeline model_id={self.model_id}"