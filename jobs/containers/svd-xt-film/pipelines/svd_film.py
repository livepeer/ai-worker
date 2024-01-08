import torch
from diffusers import StableVideoDiffusionPipeline
from pipelines.film import FILMPipeline
from pipelines.util import ListReader, VideoWriter
from PIL import Image
import einops
from typing import Union, List
import os


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
        image: Union[str, List[str]],
        motion_bucket_id: float = 127,
        noise_aug_strength: float = 0.02,
        inter_frames: int = 2,
    ):
        generator = torch.manual_seed(42)

        if isinstance(image, str):
            batch_size = 1
            input_image = Image.open(image).convert("RGB")
        else:
            batch_size = len(image)
            input_image = [Image.open(i).convert("RGB") for i in image]

        batch_frames = self.svd_xt_pipeline(
            input_image,
            decode_chunk_size=8,
            generator=generator,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            output_type="np",
        ).frames

        # batch_frames is a List[np.ndarray] (ideally it would be np.ndarray)
        # Each np.ndarray in the list has the shape f h w c i.e. (25, 576, 1024, 3)

        batch_frames = torch.stack(
            [torch.from_numpy(frames) for frames in batch_frames]
        )
        batch_frames = einops.rearrange(batch_frames, "n f h w c -> n f c h w")

        # 12 fps for 25 frames -> ~2s video
        fps = 12.0

        if inter_frames > 0:
            tot_frames = 25 + (25 // 2) * inter_frames
            fps = tot_frames // 2

            # TODO: Refactor FILMPipeline so it can accept a batch instead of reading frame by frame
            output_frames = []
            for i in range(batch_size):
                frames = batch_frames[i]
                reader = ListReader(frames)
                frames = self.film_pipeline(reader, inter_frames=inter_frames)
                output_frames.append(einops.rearrange(frames, "b 1 c h w -> b c h w"))

            batch_frames = einops.rearrange(output_frames, "b f c h w -> b f c h w")

        if output_path is not None:
            for i in range(batch_size):
                frames = batch_frames[i]
                reader = ListReader(frames)
                height, width = reader.get_resolution()
                writer = VideoWriter(
                    output_path=os.path.join(output_path, f"{i}_out.mp4"),
                    height=height,
                    width=width,
                    fps=fps,
                    format="rgb24",
                )

                writer.open()

                while True:
                    frame = reader.get_frame()
                    if frame is None:
                        break

                    writer.write_frame(frame)

                writer.close()
