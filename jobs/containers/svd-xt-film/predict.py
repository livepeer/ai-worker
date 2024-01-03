from cog import BasePredictor, Input, Path
from diffusers import StableVideoDiffusionPipeline
import torch
from PIL import Image
from pipelines.film import FILMPipeline
from pipelines.util import DirectoryReader, DirectoryWriter, VideoWriter
import os


class Predictor(BasePredictor):
    def setup(self):
        repo_id = "stabilityai/stable-video-diffusion-img2vid-xt"
        self.svd_xt_pipeline = StableVideoDiffusionPipeline.from_pretrained(
            repo_id, torch_dtype=torch.float16, variant="fp16", cache_dir="./cache"
        )
        self.svd_xt_pipeline.enable_model_cpu_offload()

        model_path = "./cache/film_net_fp16.pt"
        self.film_pipeline = FILMPipeline(model_path)
        self.film_pipeline = self.film_pipeline.to("cuda", torch.float16)

    def predict(
        self,
        image: Path = Input(description="Image to guide generation"),
        motion_bucket_id: float = Input(
            description="The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video",
            default=127,
        ),
        noise_aug_strength: float = Input(
            description="The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion",
            default=0.02,
        ),
    ) -> Path:
        generator = torch.manual_seed(42)
        frames = self.svd_xt_pipeline(
            Image.open(image),
            decode_chunk_size=8,
            generator=generator,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            output_type="np",
        ).frames[0]

        svd_xt_output_dir = "svd_xt_output"
        svd_xt_writer = DirectoryWriter(svd_xt_output_dir)
        for frame in frames:
            svd_xt_writer.write_frame(torch.from_numpy(frame))

        film_reader = DirectoryReader(svd_xt_output_dir)
        height, width = film_reader.get_resolution()

        # svd-xt outputs 25 frames by default
        # If we generate 2 intermediate frames for each pair of frames, we will have 24 intermediate frames
        # 25 + 24 = 49 frames
        # At fps = 24.0, this will be a ~2 second video
        fps = 24.0
        inter_frames = 2

        film_output_path = os.path.join("film_output", "output.mp4")
        film_writer = VideoWriter(
            output_path=film_output_path,
            height=height,
            width=width,
            fps=fps,
            format="rgb24",
        )

        self.film_pipeline(film_reader, film_writer, inter_frames=inter_frames)

        # TODO: Ensure that these temp files are cleared
        return Path(film_output_path)
