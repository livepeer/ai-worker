from cog import BasePredictor, Input, File, Path
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
import torch

class Predictor(BasePredictor):
  def setup(self):
    local_repo = "checkpoints"
    self.pipeline = StableVideoDiffusionPipeline.from_pretrained(local_repo, torch_dtype=torch.float16, variant="fp16")
    self.pipeline.enable_model_cpu_offload()

  def predict(
    self,
    image: File = Input(description="Image to guide generation"),
    motion_bucket_id: float = Input(description="The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video", default=127),
    noise_aug_strength: float = Input(description="The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion", default=0.02)
  ) -> Path:
    generator = torch.manual_seed(42)
    frames = self.pipeline(image, decode_chunk_size=8, generator=generator, motion_bucket_id=motion_bucket_id, noise_aug_strength=noise_aug_strength).images[0]
    # TODO: Ensure that these temp files are cleared
    export_to_video(frames, "generated.mp4", fps=7)
    return Path("generated.mp4")