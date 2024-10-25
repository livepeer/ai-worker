import logging
import os
import uuid

import soundfile as sf
import torch
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TextToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.device = get_torch_device()
        self.model_id = model_id
        kwargs = {"cache_dir": get_model_dir()}

        self.TTS_model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id,
            **kwargs,
        ).to(self.device)

        self.TTS_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2", # TODO (pschroedl): investigate installing flash_attn + cuda toolkit in Dockerfile.text_to_speech
            **kwargs,
        )

    def __call__(self, params):
        # generate unique filename
        unique_audio_filename = f"{uuid.uuid4()}.wav"
        audio_path = os.path.join("/tmp/", unique_audio_filename)
        self.generate_speech(params.text_input, params.description, audio_path)
        return audio_path

    def generate_speech(self, text, tts_steering, output_file_name):
        with torch.no_grad():
            input_ids = self.TTS_tokenizer(
                tts_steering, return_tensors="pt"
            ).input_ids.to(self.device)
            prompt_input_ids = self.TTS_tokenizer(
                text, return_tensors="pt"
            ).input_ids.to(self.device)

            generation = self.TTS_model.generate(
                input_ids=input_ids, prompt_input_ids=prompt_input_ids
            )
            generated_audio = generation.cpu().numpy().squeeze()

            sf.write(output_file_name, generated_audio, samplerate=44100)

            # Free the tensors
            del input_ids, prompt_input_ids, generation, generated_audio

        return output_file_name

    def __str__(self) -> str:
        return f"Text-To-Speech model_id={self.model_id}"
