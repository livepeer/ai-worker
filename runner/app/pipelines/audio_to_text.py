import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List

import torch
import warnings
from fastapi import File, UploadFile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from app.pipelines.utils.audio import AudioConverter
from app.utils.errors import InferenceError

logger = logging.getLogger(__name__)


class ModelName(Enum):
    """Enumeration mapping model names to their corresponding IDs. Returns None if the
    model ID is not found."""

    WHISPER_LARGE_V3 = "openai/whisper-large-v3"
    WHISPER_MEDIUM = "openai/whisper-medium"
    WHISPER_DISTIL_LARGE_V3 = "distil-whisper/distil-large-v3"

    @classmethod
    def list(cls):
        """Return a list of all model IDs."""
        return [model.value for model in cls]

    @classmethod
    def from_value(cls, value: str) -> Enum | None:
        """Return the enum member corresponding to the given value, or None if not
        found."""
        try:
            return cls(value)
        except ValueError:
            return None


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    torch_dtype: torch.dtype = (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    chunk_length_s: int = 30
    batch_size: int = 16


MODEL_CONFIGS = {
    ModelName.WHISPER_LARGE_V3: ModelConfig(),
    ModelName.WHISPER_MEDIUM: ModelConfig(torch_dtype=torch.float32),
    ModelName.WHISPER_DISTIL_LARGE_V3: ModelConfig(chunk_length_s=25),
}
INCOMPATIBLE_EXTENSIONS = ["mp4", "m4a", "ac3"]


class AudioToTextPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {}

        torch_device = get_torch_device()

        # Get the GPU device properties
        device = torch.cuda.get_device_properties(0)
        major, minor = device.major, device.minor

        # Compute capability of Ampere starts at 8.0
        if (major, minor) >= (8, 0):
            attn_implementation = "flash_attention_2"
        else:
            warnings.warn(f"GPU {device.name} (Compute Capability {major}.{minor}) is not compatible with FlashAttention so, scaled_dot_product_attention is being used instead")
            attn_implementation = "sdpa"

        # Get model specific configuration parameters.
        model_enum = ModelName.from_value(model_id)
        self._model_cfg: ModelConfig = MODEL_CONFIGS.get(model_enum, ModelConfig())
        kwargs["torch_dtype"] = self._model_cfg.torch_dtype
        logger.info(
            "AudioToText loading '%s' on device '%s' with '%s' variant",
            model_id,
            torch_device,
            kwargs["torch_dtype"],
        )

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=get_model_dir(),
            torch_dtype="auto",
            attn_implementation=attn_implementation,
            **kwargs,
        ).to(torch_device)

        processor = AutoProcessor.from_pretrained(model_id, cache_dir=get_model_dir())

        self.tm = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            device=torch_device,
            **kwargs,
        )

        self._audio_converter = AudioConverter()

    def __call__(self, audio: UploadFile, duration: float, **kwargs) -> List[File]:
        audioBytes = audio.file.read()

        # Convert M4A/MP4 files for pipeline compatibility.
        if (
            os.path.splitext(audio.filename)[1].lower().lstrip(".")
            in INCOMPATIBLE_EXTENSIONS
        ):
            audioBytes = self._audio_converter.convert(audioBytes, "mp3")

        # Adjust batch size and chunk length based on timestamps and duration.
        # NOTE: Done to prevent CUDA OOM errors for large audio files.
        kwargs["batch_size"] = self._model_cfg.batch_size
        kwargs["chunk_length_s"] = self._model_cfg.chunk_length_s
        if kwargs["return_timestamps"] == "word":
            if duration > 3600:
                raise InferenceError(
                    f"Word timestamps are only supported for audio files up to 60 minutes for model {self.model_id}"
                )
            if duration > 200:
                kwargs["batch_size"] = 4
        if duration <= kwargs["chunk_length_s"]:
            kwargs.pop("batch_size", None)
            kwargs.pop("chunk_length_s", None)
            inference_mode = "sequential"
        else:
            inference_mode = f"chunked (batch_size={kwargs['batch_size']}, chunk_length_s={kwargs['chunk_length_s']})"
        logger.info(
            f"AudioToTextPipeline: Starting inference mode={inference_mode} with duration={duration}"
        )

        try:
            outputs = self.tm(audioBytes, **kwargs)
            outputs.setdefault("chunks", [])
        except torch.cuda.OutOfMemoryError as e:
            raise e
        except Exception as e:
            raise InferenceError(original_exception=e)

        return outputs

    def __str__(self) -> str:
        return f"AudioToTextPipeline model_id={self.model_id}"
