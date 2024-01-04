import torch
from diffusers import StableVideoDiffusionPipeline
from pipelines.film import FILMPipeline
from pipelines.util import ListReader, VideoWriter
from PIL import Image
import einops


class StableVideoDiffusionFILMPipeline:
    def __init__(
        self,
        cache_dir: str,
        svd_config: dict = {
            "sfast": False,
            "quantize": False,
            "no_fusion": False,
        },
    ):
        repo_id = "stabilityai/stable-video-diffusion-img2vid-xt"
        self.svd_xt_pipeline = StableVideoDiffusionPipeline.from_pretrained(
            repo_id, cache_dir=cache_dir, variant="fp16", torch_dtype=torch.float16
        )
        self.svd_xt_pipeline = self.svd_xt_pipeline.to("cuda")

        if svd_config["quantize"]:
            from diffusers.utils import USE_PEFT_BACKEND

            assert USE_PEFT_BACKEND
            self.svd_xt_pipeline.unet = torch.quantization.quantize_dynamic(
                self.svd_xt_pipeline.unet,
                {torch.nn.Linear},
                dtype=torch.qint8,
                inplace=True,
            )

        if svd_config["no_fusion"]:
            torch.jit.set_fusion_strategy([("STATIC", 0), ("DYNAMIC", 0)])

        if svd_config["sfast"]:
            from pipelines.sfast import compile_model

            self.svd_xt_pipeline = compile_model(self.svd_xt_pipeline)

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
