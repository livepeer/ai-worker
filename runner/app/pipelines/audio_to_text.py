from enum import Enum
import logging
import os
from typing import List

import torch
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from app.pipelines.utils.audio import AudioConverter
from app.utils.errors import InferenceError
from fastapi import File, UploadFile
from huggingface_hub import file_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logger = logging.getLogger(__name__)

class ModelName(Enum):
    """Enumeration mapping model names to their corresponding IDs."""

    WHISPER_LARGE_V3 = "openai/whisper-large-v3"
    WHISPER_MEDIUM = "openai/whisper-medium"
    WHISPER_DISTIL_LARGE_V3 = "distil-whisper/distil-large-v3"

    @classmethod
    def list(cls):
        """Return a list of all model IDs."""
        return list(map(lambda c: c.value, cls))
    
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

MODEL_INCOMPATIBLE_EXTENSIONS = {
    ModelName.WHISPER_LARGE_V3: ["mp4", "m4a", "ac3"],
    ModelName.WHISPER_MEDIUM: ["mp4", "m4a", "ac3"],
    ModelName.WHISPER_DISTIL_LARGE_V3: ["mp4", "m4a", "ac3"]
}

class AudioToTextPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {}

        torch_device = get_torch_device()

        # Retrieve torch_dtype from MODEL_OPT_DEFAULTS
        model_type = ModelName(model_id)
        kwargs["torch_dtype"] = MODEL_OPT_DEFAULTS[model_type]["torch_dtype"]
        logger.info("AudioToText loading %s on device %s with %s variant", model_id, torch_device, kwargs["torch_dtype"])

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=get_model_dir(),
            attn_implementation="eager",
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

        self.audio_converter = AudioConverter()

    def __call__(self, audio: UploadFile, duration: float, **kwargs) -> List[File]:
        model_type = ModelName(self.model_id)
        audioBytes = audio.file.read()

        # Convert M4A/MP4 files for pipeline compatibility.
        if (
            os.path.splitext(audio.filename)[1].lower().lstrip(".")
            in MODEL_INCOMPATIBLE_EXTENSIONS[model_type]
        ):
            audioBytes = self.audio_converter.convert(audioBytes, "mp3")

        chunk_length_s = int(MODEL_OPT_DEFAULTS[model_type].get("chunk_length_s")) 
        batch_size = int(16)
        kwargs["batch_size"] = batch_size
        kwargs["chunk_length_s"] = chunk_length_s
        
        # if word timestamps are requested, then reduce batch size to 4
        if (kwargs["return_timestamps"] == 'word'):
            # Maximum 60 minute audio file length for return_timestamps='word'
            word_timestamp_max_s = 3600
            if duration > word_timestamp_max_s:
                raise InferenceError("Word timestamps are only supported for audio files up to 60 minutes long for model %s" % (self.model_id))
            
            #optimized defaults for word timestamps
            if duration > 300:
                kwargs["batch_size"] = 4

        # if duration is short then use sequential inference
        if duration <= chunk_length_s:
            kwargs.pop("batch_size", None)
            kwargs.pop("chunk_length_s", None)

        try:
            logger.info(
                "AudioToTextPipeline: Starting inference with batch_size=%s, chunk_length_s=%s, duration=%s",
                kwargs.get("batch_size", "N/A"),
                kwargs.get("chunk_length_s", "N/A"),
                duration
            )

            outputs = self.tm(audioBytes, **kwargs)
            outputs.setdefault("chunks", [])

        except torch.cuda.OutOfMemoryError as e:
            logger.error("CUDA Out of Memory Error during inference")
            raise e
        except Exception as e:
            raise InferenceError(original_exception=e)

        return outputs

    def __str__(self) -> str:
        return f"AudioToTextPipeline model_id={self.model_id}"
