import torch
import numpy as np
import os
import shutil
import time
import sys
sys.path.append('C://Users//ganes//ai-worker//runner')
from PIL import PngImagePlugin, Image
from diffusers import StableVideoDiffusionPipeline
from app.pipelines.frame_interpolation import FILMPipeline
from app.pipelines.utils import DirectoryReader, frames_compactor, DirectoryWriter

# Increase the max text chunk size for PNG images
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

def move_file_with_retry(src_file_path: str, dst_file_path: str, retries: int = 5, delay: float = 1.0):
    for attempt in range(retries):
        try:
            shutil.move(src_file_path, dst_file_path)
            return
        except PermissionError:
            print(f"Attempt {attempt + 1} failed: File is in use. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise PermissionError(f"Failed to move file after {retries} attempts.")

def get_last_file_sorted_by_name(directory: str) -> str:
    """
    Get the last file in the directory when sorted by filename.

    Args:
        directory (str): Path to the directory.

    Returns:
        str: Path to the last file in the sorted list, or None if directory is empty.
    """
    try:
        # List all files in the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        if not files:
            print("No files found in the directory.")
            return None
        
        # Sort files by name
        files.sort()
        
        # Get the last file in the sorted list
        last_file = files[-1]
        
        return os.path.join(directory, last_file)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    # Initialize pipelines
    repo_id = "stabilityai/stable-video-diffusion-img2vid-xt"
    svd_xt_pipeline = StableVideoDiffusionPipeline.from_pretrained(
        repo_id, torch_dtype=torch.float16, variant="fp16", cache_dir="G:/ai-models/models"
    )
    svd_xt_pipeline.enable_model_cpu_offload()
    
    film_pipeline = FILMPipeline("G:/ai-models/models/film_net_fp16.pt").to(device="cuda", dtype=torch.float16)

    # Load initial input image
    image_path = "G:/ai-models/models/gif_frames/donut_motion.png"
    image = Image.open(image_path)
    
    fps = 24.0
    inter_frames = 4
    rounds = 2  # Number of rounds of generation and interpolation
    base_output_dir = "G:/ai-models/models"

    all_frames_dir = os.path.join(base_output_dir, "all_interpolated_frames")
    os.makedirs(all_frames_dir, exist_ok=True)
    
    last_frame_for_next_round = os.path.join(base_output_dir, "last_frame_for_next_round.png")
    
    for round_num in range(1, rounds + 1):
        svd_xt_output_dir = os.path.join(base_output_dir, f"svd_xt_output_round_{round_num}")
        os.makedirs(svd_xt_output_dir, exist_ok=True)
        
        # Generate frames using SVD pipeline
        generator = torch.manual_seed(42)
        frames = svd_xt_pipeline(image, decode_chunk_size=8, generator=generator, output_type="np").frames[0]
        
        # Save SVD frames to directory
        film_writer = DirectoryWriter(svd_xt_output_dir)
        for idx, frame in enumerate(frames):
            film_writer.write_frame(torch.tensor(frame).permute(2, 0, 1))
        
        # Read saved frames for interpolation
        film_reader = DirectoryReader(svd_xt_output_dir)
        height, width = film_reader.get_resolution()
        
        # Interpolate frames using FILM pipeline
        film_pipeline(film_reader, film_writer, inter_frames=inter_frames)

        # Close reader and writer
        film_writer.close()
        film_reader.reset()

        # Deleting the SVD generated images.
        for i in range(len(frames)):
            os.remove(os.path.join(svd_xt_output_dir, f"{i}.png"))
        print(f"Deleted the first {len(frames)} frames in directory: {svd_xt_output_dir}")

        # Save the last frame separately for the next round
        last_frame_path = get_last_file_sorted_by_name(svd_xt_output_dir)
        if last_frame_path:
            shutil.copy2(last_frame_path, last_frame_for_next_round)
        else:
            print("No frames found to copy.")        

        # Move all interpolated frames to a common directory with a unique naming scheme
        for file_name in sorted(os.listdir(svd_xt_output_dir)):
            src_file_path = os.path.join(svd_xt_output_dir, file_name)
            dst_file_name = f"round_{round_num:03d}_frame_{file_name}"
            dst_file_path = os.path.join(all_frames_dir, dst_file_name)
            
            move_file_with_retry(src_file_path, dst_file_path)
        
        # Clean up the source directory after moving frames
        for file_name in os.listdir(svd_xt_output_dir):
            os.remove(os.path.join(svd_xt_output_dir, file_name))
        os.rmdir(svd_xt_output_dir)
        
        # Ensure all operations on last frame are complete before opening it again
        time.sleep(1)  # Small delay to ensure file system operations are complete
        
        # Prepare for next round
        image = Image.open(last_frame_for_next_round)
    
    # Compile all interpolated frames from all rounds into a final video
    video_output_dir = "G:/ai-models/models/video_out"
    os.makedirs(video_output_dir, exist_ok=True)
    
    film_output_path = os.path.join(video_output_dir, "output.avi")
    frames_compactor(frames=all_frames_dir, output_path=film_output_path, fps=fps, codec="MJPG", is_directory=True)
    
    # Clean up all frames in the directories after video generation
    for file_name in os.listdir(all_frames_dir):
        file_path = os.path.join(all_frames_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os.rmdir(all_frames_dir)
    
    print(f"All frames deleted from directories.")
    print(f"Video generated at: {film_output_path}")
    return film_output_path

if __name__ == "__main__":
    main()
