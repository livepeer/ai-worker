import uuid
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_torch_device, get_model_dir
from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel, FastSpeech2ConformerHifiGan
from huggingface_hub import file_download
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
        self.TTS_tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer", cache_dir=get_model_dir())
        self.TTS_model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer", cache_dir=get_model_dir()).to(self.device)
        self.TTS_hifigan = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan", cache_dir=get_model_dir()).to(self.device)

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
        inputs = self.TTS_tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"].to(self.device)
        output_dict = self.TTS_model(input_ids, return_dict=True)
        spectrogram = output_dict["spectrogram"]
        waveform = self.TTS_hifigan(spectrogram)
        sf.write(output_file_name, waveform.squeeze().detach().cpu().numpy(), samplerate=22050)
        return output_file_name
    
    def __str__(self) -> str:
        return f"TextToSpeechPipeline model_id={self.model_id}"
    