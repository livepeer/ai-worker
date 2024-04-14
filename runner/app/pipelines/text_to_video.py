from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir

from diffusers import AnimateDiffPipeline, MotionAdapter, DiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from huggingface_hub import file_download, hf_hub_download
from safetensors.torch import load_file
import torch
import PIL
from typing import List
import logging
import os

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# See https://huggingface.co/ByteDance/AnimateDiff-Lightning
# Compatible with any SD1.5 stylized base model
bases = {
    "AbsoluteReality": "digiplay/AbsoluteReality_v1.8.1",
    "epiCRealism": "emilianJR/epiCRealism",
    "DreamShaper": "Lykon/DreamShaper",
    "RealisticVision": "SG161222/Realistic_Vision_V6.0_B1_noVAE"
}
base_loaded = "epiCRealism"

motionChoices = [
    ("Default", ""),
    ("Zoom in", "guoyww/animatediff-motion-lora-zoom-in"),
    ("Zoom out", "guoyww/animatediff-motion-lora-zoom-out"),
    ("Tilt up", "guoyww/animatediff-motion-lora-tilt-up"),
    ("Tilt down", "guoyww/animatediff-motion-lora-tilt-down"),
    ("Pan left", "guoyww/animatediff-motion-lora-pan-left"),
    ("Pan right", "guoyww/animatediff-motion-lora-pan-right"),
    ("Roll left", "guoyww/animatediff-motion-lora-rolling-anticlockwise"),
    ("Roll right", "guoyww/animatediff-motion-lora-rolling-clockwise"),
],

logger = logging.getLogger(__name__)
torch.backends.cuda.matmul.allow_tf32 = True

class TextToVideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        has_fp16_variant = any(
            ".fp16.safetensors" in fname
            for _, _, files in os.walk(folder_path)
            for fname in files
        )
        if torch_device != "cpu" and has_fp16_variant:
            logger.info("TextToVideoPipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"
            kwargs["use_safetensors"] = True

        self.model_id = model_id

        if self.model_id == "ByteDance/AnimateDiff-Lightning":
            kwargs["torch_dtype"] = torch.float16
            repo = "ByteDance/AnimateDiff-Lightning"
            self.motion_loaded = ""
            # Load base
            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
            self.ldm = AnimateDiffPipeline.from_pretrained(bases[base_loaded], motion_adapter=adapter, torch_dtype=torch.float16).to("cuda")
            # Load noise scheduler
            self.ldm.scheduler = EulerDiscreteScheduler.from_config(self.ldm.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
            # Load model params
            ckpt = "animatediff_lightning_4step_diffusers.safetensors"
            self.ldm.unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"), strict=False)
        elif self.model_id == "ali-vilab/text-to-video-ms-1.7b":
            self.ldm = DiffusionPipeline.from_pretrained(model_id, **kwargs)
            self.ldm.scheduler = DPMSolverMultistepScheduler.from_config(self.ldm.scheduler.config)
            self.ldm.to(torch_device)
        else:
            self.ldm = DiffusionPipeline.from_pretrained(model_id, **kwargs).to(torch_device)
        self.ldm.enable_vae_slicing()

        if os.environ.get("SFAST"):
            logger.info(
                "TextToVideoPipeline will be dynamicallly compiled with stable-fast for %s",
                model_id,
            )
            from app.pipelines.sfast import compile_model

            self.ldm = compile_model(self.ldm)

    def __call__(self, prompt: str, **kwargs) -> List[List[PIL.Image]]:
        # ali-vilab/text-to-video-ms-1.7b has a limited parameter set
        if self.model_id == "ali-vilab/text-to-video-ms-1.7b":
            kwargs["num_inference_steps"] = 25
            if "fps" in kwargs:
                del kwargs["fps"]
        elif self.model_id == "ByteDance/AnimateDiff-Lightning":
            # Load correct motion module
            if self.motion_loaded != kwargs["motion"]:
                self.ldm.unload_lora_weights()
                if kwargs["motion"] != "":
                    self.ldm.load_lora_weights(kwargs["motion"], adapter_name="motion")
                    self.ldm.set_adapters(["motion"], [0.7])
                self.motion_loaded = kwargs["motion"]
            # This model is finetuned for 2, 4 or 8 inference steps
            kwargs["num_inference_steps"] = 4
            kwargs["guidance_scale"] = 1
            kwargs["num_frames"] = 18
            if "fps" in kwargs:
                kwargs["frame_rate"] = kwargs["fps"]
                del kwargs["fps"]

        seed = kwargs.pop("seed", None)
        if seed is not None:
            if isinstance(seed, int):
                kwargs["generator"] = torch.Generator(get_torch_device()).manual_seed(
                    seed
                )
            elif isinstance(seed, list):
                kwargs["generator"] = [
                    torch.Generator(get_torch_device()).manual_seed(s) for s in seed
                ]

        return self.ldm(prompt, **kwargs).frames

    def __str__(self) -> str:
        return f"TextToVideoPipeline model_id={self.model_id}"
