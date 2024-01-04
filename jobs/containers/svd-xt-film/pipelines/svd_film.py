import torch
from diffusers import StableVideoDiffusionPipeline
from pipelines.film import FILMPipeline
from pipelines.util import ListReader, VideoWriter
from PIL import Image
import einops


class StableVideoDiffusionFILMPipeline:
    def __init__(self, cache_dir: str):
        repo_id = "stabilityai/stable-video-diffusion-img2vid-xt"
        self.svd_xt_pipeline = StableVideoDiffusionPipeline.from_pretrained(
            repo_id, torch_dtype=torch.float16, variant="fp16", cache_dir=cache_dir
        )
        self.svd_xt_pipeline.enable_model_cpu_offload()

        self.film_pipeline = FILMPipeline(f"{cache_dir}/film_net_fp16.pt")
        self.film_pipeline = self.film_pipeline.to(device="cuda", dtype=torch.float16)

    def __call__(
        self,
        output_path: str,
        image: str,
        motion_bucket_id: float = 127,
        noise_aug_strength: float = 0.02,
    ):
        generator = torch.manual_seed(42)

        frames = self.svd_xt_pipeline(
            Image.open(image).convert("RGB"),
            decode_chunk_size=8,
            generator=generator,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            output_type="np",
        ).frames[0]

        frames = [torch.from_numpy(frame) for frame in frames]
        frames = einops.rearrange(frames, "n h w c -> n c h w")

        # svd-xt outputs 25 frames by default
        # If we generate 2 intermediate frames for each pair of frames, we will have 24 intermediate frames
        # 25 + 24 = 49 frames
        # At fps = 24.0, this will be a ~2 second video
        fps = 24.0
        inter_frames = 2

        reader = ListReader(frames)
        height, width = reader.get_resolution()
        writer = VideoWriter(
            output_path=output_path,
            height=height,
            width=width,
            fps=fps,
            format="rgb24",
        )

        self.film_pipeline(reader, writer, inter_frames=inter_frames)
