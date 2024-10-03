from enum import Enum
import logging
import os
from typing import List

import torch
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from app.pipelines.utils.audio import AudioConverter
from fastapi import File, UploadFile
from huggingface_hub import file_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logger = logging.getLogger(__name__)

MODEL_INCOMPATIBLE_EXTENSIONS = {
    "openai/whisper-large-v3": ["mp4", "m4a", "ac3"],
    "openai/whisper-medium": ["mp4", "m4a", "ac3"],
    "distil-whisper/distil-large-v3": ["mp4", "m4a", "ac3"]
}

class ModelName(Enum):
    """Enumeration mapping model names to their corresponding IDs."""

    WHISPER_LARGE_V3 = "openai/whisper-large-v3"
    WHISPER_MEDIUM = "openai/whisper-medium"
    WHISPER_DISTIL_LARGE_V3 = "distil-whisper/distil-large-v3"

    @classmethod
    def list(cls):
        """Return a list of all model IDs."""
        return list(map(lambda c: c.value, cls))

class AudioToTextPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {}

        torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)

        MODEL_OPT_DEFAULTS = {
            ModelName.WHISPER_LARGE_V3: {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "chunk_length_s": 30
            },
            ModelName.WHISPER_MEDIUM: 
            {
                "torch_dtype": torch.float32,
                "chunk_length_s": 30
            },
            ModelName.WHISPER_DISTIL_LARGE_V3:
            {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "chunk_length_s": 25
            }
        }
        
        # Map model_id to ModelName enum
        model_name_enum = next(key for key, value in ModelName.__members__.items() if value.value == model_id)
        model_type = ModelName[model_name_enum]

        # Retrieve torch_dtype from MODEL_OPT_DEFAULTS
        kwargs["torch_dtype"] = MODEL_OPT_DEFAULTS[model_type].get("torch_dtype", torch.float16)

        if torch_device != "cpu" and kwargs["torch_dtype"] == torch.float16:
            logger.info("AudioToText loading %s variant for fp16", model_id)
            
        elif torch_device != "cpu" and kwargs["torch_dtype"] == torch.float32:
            logger.info("AudioToText loading %s variant for f32", model_id)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=get_model_dir(),
            **kwargs,
        ).to(torch_device)

        # TODO: If audio duration is greater than 30 seconds, split the audio into chunks of 30 seconds.
        kwargs["chunk_length_s"] = MODEL_OPT_DEFAULTS[model_type].get("chunk_length_s", 30)
        kwargs["batch_size"] = 16

        processor = AutoProcessor.from_pretrained(model_id, cache_dir=get_model_dir())

        self.tm = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            return_timestamps=True,
            **kwargs,
        )

    def __call__(self, audio: UploadFile, **kwargs) -> List[File]:
        # Convert M4A/MP4 files for pipeline compatibility.
        if (
            os.path.splitext(audio.filename)[1].lower().lstrip(".")
            in MODEL_INCOMPATIBLE_EXTENSIONS[self.model_id]
        ):
            audio_converter = AudioConverter()
            converted_bytes = audio_converter.convert(audio, "mp3")
            audio_converter.write_bytes_to_file(converted_bytes, audio)

        return self.tm(audio.file.read(), **kwargs)

    def __str__(self) -> str:
        return f"AudioToTextPipeline model_id={self.model_id}"
