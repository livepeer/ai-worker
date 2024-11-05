from enum import Enum
import logging
import os
from typing import List
from dataclasses import dataclass

import torch
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from app.pipelines.utils.audio import AudioConverter
from app.utils.errors import InferenceError
from fastapi import File, UploadFile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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
    def get(cls, model_id: str) -> Enum | None:
        """Return the enum or None if the model ID is not found."""
        try:
            return cls(model_id)
        except ValueError:
            return None


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    torch_dtype: torch.dtype = (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    chunk_length_s: int = 30


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

        # Get model specific configuration parameters.
        model_enum = ModelName.get(model_id)
        self._model_cfg = (
            ModelConfig() if model_enum is None else MODEL_CONFIGS[model_enum]
        )
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
            attn_implementation="eager",  # TODO: enable flash attention.
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
        kwargs["batch_size"] = 16
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
        logger.info(
            f"AudioToTextPipeline: Starting inference with batch_size={kwargs.get('batch_size', 'N/A')}, "
            f"chunk_length_s={kwargs.get('chunk_length_s', 'N/A')}, duration={duration}"
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
