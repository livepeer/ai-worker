import uuid
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_torch_device, get_model_dir
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import file_download
import torch
import soundfile as sf
import os
import logging

logger = logging.getLogger(__name__)

class TextToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        if os.getenv("MOCK_PIPELINE", "").strip().lower() == "true":
            logger.info("Mocking TextToSpeechPipeline for %s", model_id)
            return

        self.device = get_torch_device()

        self.model = AutoModelForSeq2SeqLM.from_pretrained("parler-tts/parler-tts-large-v1")
        self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

        # # compile the forward pass
        # compile_mode = "default" # chose "reduce-overhead" for 3 to 4x speed-up
        # self.model.generation_config.cache_implementation = "static"
        # self.model.forward = torch.compile(self.model.forward, mode=compile_mode)

        # # warmup
        # inputs = self.tokenizer("This is for compilation", return_tensors="pt", padding="max_length", max_length=max_length).to(self.device)

        # model_kwargs = {**inputs, "prompt_input_ids": inputs.input_ids, "prompt_attention_mask": inputs.attention_mask, }

        # n_steps = 1 if compile_mode == "default" else 2
        # for _ in range(n_steps):
        #     _ = self.model.generate(**model_kwargs)



    def __call__(self, text):
        if os.getenv("MOCK_PIPELINE", "").strip().lower() == "true":
            unique_audio_filename = f"{uuid.uuid4()}.wav"
            audio_path = os.path.join("/tmp/", unique_audio_filename)
            sf.write(audio_path, [0] * 22050, samplerate=22050)
            return audio_path
        unique_audio_filename = f"{uuid.uuid4()}.wav"
        audio_path = os.path.join("/tmp/", unique_audio_filename)

        self.generate_audio(text, audio_path)

        return audio_path

    def generate_audio(self, text, output_file_name):
        description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write("parler_tts_out.wav", audio_arr, self.model.config.sampling_rate)
        return output_file_name
    
    def __str__(self) -> str:
        return f"TextToSpeechPipeline model_id={self.model_id}"
    