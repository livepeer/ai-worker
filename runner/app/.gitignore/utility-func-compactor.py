import subprocess
import numpy as np
import torch
import os
from typing import List, Union
from pathlib import Path
import cv2

def generate_video_from_frames(
    frames: Union[List[np.ndarray], List[torch.Tensor]], 
    output_path: str, 
    fps: float, 
    codec: str = "MJPEG",
    is_directory: bool = False,
    width: int = None,
    height: int = None
) -> None:
    """
    Generate a video from a list of frames. Frames can be from a directory or in-memory.
    
    Args:
        frames (List[np.ndarray] | List[torch.Tensor]): List of frames as NumPy arrays or PyTorch tensors.
        output_path (str): Path to save the output video file.
        fps (float): Frames per second for the video.
        codec (str): Codec used for video compression (default is "MJPEG").
        is_directory (bool): If True, treat `frames` as a directory path containing image files.
        width (int): Width of the video. Must be provided if `frames` are in-memory.
        height (int): Height of the video. Must be provided if `frames` are in-memory.
    
    Returns:
        None
    """
    if is_directory:
        # Read frames from a directory
        frames = [cv2.imread(os.path.join(frames, file)) for file in sorted(os.listdir(frames))]
    else:
        # Convert torch tensors to numpy arrays if necessary
        if isinstance(frames[0], torch.Tensor):
            frames = [frame.permute(1, 2, 0).cpu().numpy() for frame in frames]
        
        # Ensure frames are numpy arrays and are uint8 type
        frames = [frame.astype(np.uint8) for frame in frames]
    
    # Check if frames are consistent
    if not frames:
        raise ValueError("No frames to process.")
    
    if width is None or height is None:
        # Use dimensions of the first frame if not provided
        height, width = frames[0].shape[:2]
    
    # Write frames to a temporary directory
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(temp_dir / f"frame_{i:05d}.png"), frame)

    # Build ffmpeg command
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', str(temp_dir / 'frame_%05d.png'),
        '-c:v', codec, '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    # Run ffmpeg command
    subprocess.run(ffmpeg_cmd, check=True)

    # Clean up temporary frames
    for file in temp_dir.glob("*.png"):
        file.unlink()
    temp_dir.rmdir()

    print(f"Video saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Example with in-memory frames (as np.ndarray)
    # Assuming `in_memory_frames` is a list of numpy arrays

    # Example with frames from a directory
    frames_directory = "G:/ai-models/models/svd_xt_output"
    generate_video_from_frames(
        frames=frames_directory,
        output_path="G:/ai-models/models/video_out/output.mp4",
        fps=24.0,
        codec="mpeg4",
        is_directory=True
    )
