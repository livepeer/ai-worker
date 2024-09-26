import uuid
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_torch_device, get_model_dir
from transformers import AutoTokenizer
import torch
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import os
import torch
import logging
import time
import gc

logger = logging.getLogger(__name__)

class TextToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.device = get_torch_device()
        self.model_id = model_id
        kwargs = {"cache_dir": get_model_dir()}
        print(f"ModelID : {model_id}, cache_dir: {get_model_dir()}")
        self.TTS_model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id,
            **kwargs,
        ).to(self.device)

        self.TTS_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            **kwargs,
        )

    def __call__(self, params):
        # generate unique filename
        unique_audio_filename = f"{uuid.uuid4()}.wav"
        audio_path = os.path.join("/tmp/", unique_audio_filename)
        self.generate_speech(params.text_input, params.description, audio_path)
        return audio_path

    def generate_speech(self, text, tts_steering, output_file_name):
        try:
            with torch.no_grad():
                input_ids = self.TTS_tokenizer(tts_steering, return_tensors="pt").input_ids.to(self.device)
                prompt_input_ids = self.TTS_tokenizer(text, return_tensors="pt").input_ids.to(self.device)

                generation = self.TTS_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
                generated_audio = generation.cpu().numpy().squeeze()

                sf.write(output_file_name, generated_audio, samplerate=44100)

                # Free the tensors
                del input_ids, prompt_input_ids, generation, generated_audio
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA Out of Memory Error during speech generation.")
            self.cleanup_cuda_memory()
            raise
        finally:
            self.cleanup_cuda_memory()
        return output_file_name

    def cleanup_cuda_memory(self):
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("CUDA memory cleared.")

    def unload_model(self, model):
        # Move all components of the pipeline to CPU if they have the .cpu() method
        if hasattr(model, 'components'):
            for component in model.components.values():
                if hasattr(component, 'cpu'):
                    component.cpu()
        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(20)

    def __str__(self) -> str:
        return f"Text-To-Speech model_id={self.model_id}"