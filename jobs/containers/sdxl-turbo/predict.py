import tempfile
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionXLPipeline
import torch

class Predictor(BasePredictor):
  def setup(self):
    ckpt_file = "checkpoints/sd_xl_turbo_1.0_fp16.safetensors"
    self.pipeline = StableDiffusionXLPipeline.from_single_file(ckpt_file, torch_dtype=torch.float16)
    self.pipeline.to("cuda")

  def predict(
    self,
    prompt: str = Input(description="The prompt to guide image generation"),
    num_inference_steps: int = Input(description="The number of denoising steps", default=1)
  ) -> Path:
    # https://huggingface.co/docs/diffusers/using-diffusers/sdxl_turbo
    # guidance_scale is set to 0 because the model was trained without it
    image = self.pipeline(prompt=prompt, guidance_scale=0.0, num_inference_steps=num_inference_steps).images[0]
    # TODO: Ensure that these temp files are cleared
    output_path = Path(tempfile.mkdtemp()) / "output.png"
    image.save(output_path)
    return Path(output_path)