import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from app.utils.errors import InferenceError
from diffusers import StableAudioPipeline
from huggingface_hub import file_download
import numpy as np
import soundfile as sf
import io

logger = logging.getLogger(__name__)

class TextToAudioPipeline(Pipeline):
    def __init__(self, model_id: str):
        """Initialize the text to audio pipeline.
        
        Args:
            model_id: The model ID to use for audio generation.
        """
        self.model_id = model_id
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)

        # Load fp16 variant if available
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

        # Initialize the pipeline
        self.pipeline = StableAudioPipeline.from_pretrained(
            model_id,
            **kwargs
        ).to(torch_device)

        # Enable optimization if configured
        sfast_enabled = os.getenv("SFAST", "").strip().lower() == "true"
        if sfast_enabled:
            logger.info(
                "TextToAudioPipeline will be dynamically compiled with stable-fast for %s",
                model_id,
            )
            from app.pipelines.optim.sfast import compile_model
            self.pipeline = compile_model(self.pipeline)

    def __call__(
        self, 
        prompt: str,
        duration: float = 5.0,
        num_inference_steps: int = 10,
        guidance_scale: float = 3.0,
        negative_prompt: str = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[bytes, str]:
        """Generate audio from text.
        
        Args:
            prompt: The text prompt for audio generation.
            duration: Duration of the generated audio in seconds.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Scale for classifier-free guidance.
            negative_prompt: Optional text prompt to guide what to exclude.
            seed: Optional seed for reproducible generation.
            
        Returns:
            Tuple containing the audio data as bytes and the file format.
        """
        try:
            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                
            # Set default steps if invalid
            if num_inference_steps is None or num_inference_steps < 1:
                num_inference_steps = 10
                
            # Validate duration
            if duration < 1.0 or duration > 30.0:
                raise ValueError("Duration must be between 1 and 30 seconds")

            # Generate audio
            audio = self.pipeline(
                prompt,
                negative_prompt=negative_prompt,
                audio_length_in_s=duration,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs
            ).audio[0]
            
            # Convert to bytes using soundfile
            buffer = io.BytesIO()
            sf.write(buffer, audio, samplerate=44100, format='WAV')
            buffer.seek(0)
            
            return buffer.read(), 'wav'
            
        except Exception as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            raise InferenceError(original_exception=e)

    def __str__(self) -> str:
        return f"TextToAudioPipeline model_id={self.model_id}"