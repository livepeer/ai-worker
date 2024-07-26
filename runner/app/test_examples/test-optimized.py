import torch
import numpy as np
import os
import sys
sys.path.append('C://Users//ganes//ai-worker//runner')
from diffusers import StableVideoDiffusionPipeline
from PIL import PngImagePlugin, Image
from app.pipelines.frame_interpolation import FILMPipeline
from app.pipelines.utils import DirectoryReader, frames_compactor, DirectoryWriter

# Increase the max text chunk size for PNG images
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

def main():
    # Initialize pipelines
    repo_id = "stabilityai/stable-video-diffusion-img2vid-xt"
    svd_xt_pipeline = StableVideoDiffusionPipeline.from_pretrained(
        repo_id, torch_dtype=torch.float16, variant="fp16", cache_dir="G:/ai-models/models"
    )
    svd_xt_pipeline.enable_model_cpu_offload()
    
    film_pipeline = FILMPipeline("G:/ai-models/models/film_net_fp16.pt").to(device="cuda", dtype=torch.float16)

    # Load input image
    image_path = "G:/ai-models/models/gif_frames/rocket.png"
    image = Image.open(image_path)
    
    # Generate frames using SVD pipeline
    generator = torch.manual_seed(42)
    frames = svd_xt_pipeline(image, decode_chunk_size=8, generator=generator, output_type="np").frames[0]

    fps = 24.0
    inter_frames = 2
    svd_xt_output_dir = "G:/ai-models/models/svd_xt_output"
    video_output_dir = "G:/ai-models/models/video_out"
    
    # Save SVD frames to directory
    film_writer = DirectoryWriter(svd_xt_output_dir)
    for frame in frames:
        film_writer.write_frame(torch.tensor(frame).permute(2, 0, 1))
    
    # Read saved frames for interpolation
    film_reader = DirectoryReader(svd_xt_output_dir)
    height, width = film_reader.get_resolution()
    
    # Interpolate frames using FILM pipeline
    film_pipeline(film_reader, film_writer, inter_frames=inter_frames)
    
    # Delete original SVD frames since interpolated frames are also in the same directory.
    for i in range(len(frames)):
        os.remove(os.path.join(svd_xt_output_dir, f"{i}.png"))
    print(f"Deleted the first {len(frames)} frames in directory: {svd_xt_output_dir}")
    
    # Compile interpolated frames into a video
    film_output_path = os.path.join(video_output_dir, "output.avi")
    frames_compactor(frames=svd_xt_output_dir, output_path=film_output_path, fps=fps, codec="MJPG", is_directory=True)

    # Clean up all frames in the directory after video generation
    for file_name in os.listdir(svd_xt_output_dir):
        file_path = os.path.join(svd_xt_output_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"All frames deleted from directory: {svd_xt_output_dir}")

    print(f"Video generated at: {film_output_path}")
    return film_output_path

if __name__ == "__main__":
    main()
