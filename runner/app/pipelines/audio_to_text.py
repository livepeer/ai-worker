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


MODEL_INCOMPATIBLE_EXTENSIONS = {
    "openai/whisper-large-v3": ["mp4", "m4a", "ac3"],
}


class AudioToTextPipeline(Pipeline):
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
            logger.info("AudioToTextPipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        if os.environ.get("BFLOAT16"):
            logger.info("AudioToTextPipeline using bfloat16 precision for %s", model_id)
            kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=get_model_dir(),
            **kwargs,
        ).to(torch_device)

        processor = AutoProcessor.from_pretrained(model_id, cache_dir=get_model_dir())

        self.tm = pipeline(
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

    def __call__(self, audio: UploadFile, **kwargs) -> List[File]:
        # Convert M4A/MP4 files for pipeline compatibility.
        if (
            os.path.splitext(audio.filename)[1].lower().lstrip(".")
            in MODEL_INCOMPATIBLE_EXTENSIONS[self.model_id]
        ):
            audio_converter = AudioConverter()
            converted_bytes = audio_converter.convert(audio, "mp3")
            audio_converter.write_bytes_to_file(converted_bytes, audio)

        try:
            outputs = self.tm(audio.file.read(), **kwargs)
        except Exception as e:
            raise InferenceError(original_exception=e)

        return outputs

    def __str__(self) -> str:
        return f"AudioToTextPipeline model_id={self.model_id}"
