import logging
import io
import uuid
import torch
import torchaudio
from einops import rearrange
from fastapi import UploadFile
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

logger = logging.getLogger(__name__)

class TextToAudioPipeline(Pipeline):
    def __init__(self, model_id: str = "stabilityai/stable-audio-open-1.0"):
        self.model_id = model_id
        self.device = get_torch_device()
        
        logger.info(f"Initializing TextToAudioPipeline with model_id={model_id}")
        
        self.model, self.model_config = self.load_model()
        self.sample_rate = self.model_config["sample_rate"]
        self.sample_size = self.model_config["sample_size"]
        
        logger.info(f"Model loaded. Sample rate: {self.sample_rate}, Sample size: {self.sample_size}")

    def load_model(self):
        logger.info("Loading model...")
        model, model_config = get_pretrained_model(self.model_id)
        model = model.to(self.device)
        logger.info("Model loaded and moved to device successfully.")
        return model, model_config

    def __call__(self, prompt: str, seconds_total: int = 30, steps: int = 100, cfg_scale: float = 7) -> UploadFile:
        logger.info(f"Generating audio for prompt: '{prompt}'")
        logger.info(f"Parameters: Duration={seconds_total}s, Steps={steps}, CFG Scale={cfg_scale}")

        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": seconds_total
        }]

        output = generate_diffusion_cond(
            self.model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_size=self.sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=self.device
        )

        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        buffer = io.BytesIO()
        torchaudio.save(buffer, output, self.sample_rate, format="wav")
        buffer.seek(0)

        filename = f"generated_audio_{uuid.uuid4().hex}.wav"
        return UploadFile(filename=filename, file=buffer)

    def __str__(self) -> str:
        return f"TextToAudioPipeline model_id={self.model_id}"