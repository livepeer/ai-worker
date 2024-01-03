import torch
from pipelines.film import FILMPipeline
from pipelines.util import DirectoryReader, VideoWriter, DirectoryWriter
from diffusers import StableVideoDiffusionPipeline

from PIL import PngImagePlugin, Image

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


def main():
    repo_id = "stabilityai/stable-video-diffusion-img2vid-xt"
    svd_xt_pipeline = StableVideoDiffusionPipeline.from_pretrained(
        repo_id, torch_dtype=torch.float16, variant="fp16", cache_dir="./cache"
    )
    svd_xt_pipeline.enable_model_cpu_offload()

    film_pipeline = FILMPipeline("./cache/film_net_fp16.pt")
    film_pipeline = film_pipeline.to(device="cuda", dtype=torch.float16)

    image = "input/1.png"

    generator = torch.manual_seed(42)
    frames = svd_xt_pipeline(
        Image.open(image),
        decode_chunk_size=8,
        generator=generator,
        output_type="np",
    ).frames[0]

    svd_xt_output_dir = "svd_xt_output"
    svd_xt_writer = DirectoryWriter(svd_xt_output_dir)
    for frame in frames:
        svd_xt_writer.write_frame(torch.from_numpy(frame))

    # fps = 10
    # inter_frames = 18
    # reader = DirectoryReader("input")
    # height, width = reader.get_resolution()
    # writer = VideoWriter(
    #     output_path="output/output.mp4",
    #     height=height,
    #     width=width,
    #     fps=fps,
    #     format="rgb24",
    # )

    # pipeline(reader, writer, inter_frames=inter_frames)


if __name__ == "__main__":
    main()
