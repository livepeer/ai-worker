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
        # kwargs = {"cache_dir": get_model_dir()}

        # folder_name = file_download.repo_folder_name(
        #     repo_id=model_id, repo_type="model"
        # )
        # folder_path = os.path.join(get_model_dir(), folder_name)
        self.device = get_torch_device()
        # preload FastSpeech 2 & hifigan
        self.TTS_tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer", cache_dir=get_model_dir())
        self.TTS_model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer", cache_dir=get_model_dir()).to(self.device)
        self.TTS_hifigan = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan", cache_dir=get_model_dir()).to(self.device)


    def __call__(self, text):
        # generate unique filename
        unique_audio_filename = f"{uuid.uuid4()}.wav"
        audio_path = os.path.join("/tmp/", unique_audio_filename)

        self.generate_audio(text, audio_path)

        return audio_path

    def generate_audio(self, text, output_file_name):
        # Tokenize input text
        inputs = self.TTS_tokenizer(text, return_tensors="pt").to(self.device)
        
        # Ensure input IDs remain in Long tensor type
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generate spectrogram
        output_dict = self.TTS_model(input_ids, return_dict=True)
        spectrogram = output_dict["spectrogram"]

        # Convert spectrogram to waveform
        waveform = self.TTS_hifigan(spectrogram)

        sf.write(output_file_name, waveform.squeeze().detach().cpu().numpy(), samplerate=22050)
        return output_file_name
    
    def __str__(self) -> str:
        return f"TextToSpeechPipeline model_id={self.model_id}"
    