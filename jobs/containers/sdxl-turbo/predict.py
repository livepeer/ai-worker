import tempfile
from cog import BasePredictor, Input, Path
from diffusers import AutoPipelineForText2Image
import torch

class Predictor(BasePredictor):
  def setup(self):
    repo_id = "stabilityai/sdxl-turbo"
    self.pipeline = AutoPipelineForText2Image.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16", cache_dir="./cache")
    self.pipeline = self.pipeline.to("cuda")

  def predict(
    self,
    prompt: str = Input(description="The prompt to guide image generation"),
    num_inference_steps: int = Input(description="The number of denoising steps", default=1),
    width: int = Input(description="Width of generated image", default=1024),
    height: int = Input(description="Height of generated image", default=576)
  ) -> Path:
    # https://huggingface.co/docs/diffusers/using-diffusers/sdxl_turbo
    # guidance_scale is set to 0 because the model was trained without it
    image = self.pipeline(prompt=prompt, guidance_scale=0.0, num_inference_steps=num_inference_steps, width=width, height=height).images[0]
    # TODO: Ensure that these temp files are cleared
    output_path = Path(tempfile.mkdtemp()) / "output.png"
    image.save(output_path)
    return Path(output_path)