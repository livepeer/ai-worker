"""This module contains several utility functions."""

import logging
import os
import re
from pathlib import Path
from typing import Optional
import glob
import tempfile
from io import BytesIO
from typing import List, Union

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision.transforms import v2
import cv2
from torchaudio.io import StreamWriter
from torch import dtype as TorchDtype
from transformers import CLIPFeatureExtractor

logger = logging.getLogger(__name__)


def get_model_dir() -> Path:
    return Path(os.environ["MODEL_DIR"])


def get_model_path(model_id: str) -> Path:
    return get_model_dir() / model_id.lower()


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def validate_torch_device(device_name: str) -> bool:
    """Checks if the given PyTorch device name is valid and available.

    Args:
        device_name: Name of the device ('cuda:0', 'cuda', 'cpu').

    Returns:
        True if valid and available, False otherwise.
    """
    try:
        device = torch.device(device_name)
        if device.type == "cuda":
            # Check if CUDA is available and the specified index is within range
            if device.index is None:
                return torch.cuda.is_available()
            else:
                return device.index < torch.cuda.device_count()
        return True
    except RuntimeError:
        return False


def is_lightning_model(model_id: str) -> bool:
    """Checks if the model is a Lightning model.

    Args:
        model_id: Model ID.

    Returns:
        True if the model is a Lightning model, False otherwise.
    """
    return re.search(r"[-_]lightning", model_id, re.IGNORECASE) is not None


def is_turbo_model(model_id: str) -> bool:
    """Checks if the model is a Turbo model.

    Args:
        model_id: Model ID.

    Returns:
        True if the model is a Turbo model, False otherwise.
    """
    return re.search(r"[-_]turbo", model_id, re.IGNORECASE) is not None


def split_prompt(
    input_prompt: str,
    separator: str = "|",
    key_prefix: str = "prompt",
    max_splits: int = -1,
) -> dict[str, str]:
    """Splits an input prompt into prompts, including the main prompt, with customizable
    key naming.

    Args:
        input_prompt (str): The input prompt string to be split.
        separator (str): The character used to split the input prompt. Defaults to '|'.
        key_prefix (str): Prefix for keys in the returned dictionary for all prompts,
            including the main prompt. Defaults to 'prompt'.
        max_splits (int): Maximum number of splits to perform. Defaults to -1 (no limit).

    Returns:
        Dict[str, str]: A dictionary of all prompts, including the main prompt.
    """
    prompts = input_prompt.split(separator, max_splits - 1)
    start_index = 1 if max_splits < 0 else max(1, len(prompts) - max_splits)

    prompt_dict = {f"{key_prefix}": prompts[0].strip()}
    prompt_dict.update(
        {
            f"{key_prefix}_{i+1}": prompt.strip()
            for i, prompt in enumerate(prompts[1:], start=start_index)
        }
    )

    return prompt_dict

def frames_compactor(
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
        codec (str): Codec used for video compression (default is "XVID").
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
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to the video file
    for frame in frames:
        # Ensure each frame has the correct size
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        video_writer.write(frame)
    
    # Release the video writer
    video_writer.release()

def video_shredder(video_data, is_file_path=True) -> np.ndarray:
    """
    Extract frames from a video file or in-memory video data and return them as a NumPy array.
    
    Args:
        video_data (str or BytesIO): Path to the input video file or in-memory video data.
        is_file_path (bool): Indicates if video_data is a file path (True) or in-memory data (False).
    
    Returns:
        np.ndarray: Array of frames with shape (num_frames, height, width, channels).
    """
    if is_file_path:
        # Handle file-based video input
        video_capture = cv2.VideoCapture(video_data)
    else:
        # Handle in-memory video input
        # Create a temporary file to store in-memory video data
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_data.getvalue())
            temp_file_path = temp_file.name
        
        # Open the temporary video file
        video_capture = cv2.VideoCapture(temp_file_path)

    if not video_capture.isOpened():
        raise ValueError("Error opening video data")

    frames = []
    success, frame = video_capture.read()
    
    while success:
        frames.append(frame)
        success, frame = video_capture.read()
    
    video_capture.release()

    # Delete the temporary file if it was created
    if not is_file_path:
        os.remove(temp_file_path)
    
    # Convert list of frames to a NumPy array
    frames_array = np.array(frames)
    print(f"Extracted {frames_array.shape[0]} frames from video in shape of {frames_array.shape}")
    
    return frames_array

class SafetyChecker:
    """Checks images for unsafe or inappropriate content using a pretrained model.

    Attributes:
        device (str): Device for inference.
    """

    def __init__(
        self,
        device: Optional[str] = "cuda",
        dtype: Optional[TorchDtype] = torch.float16,
    ):
        """Initializes the SafetyChecker.

        Args:
            device: Device for inference. Defaults to "cuda".
            dtype: Data type for inference. Defaults to `torch.float16`.
        """
        device = device.lower() if device else device
        if not validate_torch_device(device):
            default_device = get_torch_device()
            logger.warning(
                f"Device '{device}' not found. Defaulting to '{default_device}'."
            )
            device = default_device

        self.device = device
        self._dtype = dtype
        self._safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(self.device)
        self._feature_extractor = CLIPFeatureExtractor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def check_nsfw_images(
        self, images: list[Image.Image]
    ) -> tuple[list[Image.Image], list[bool]]:
        """Checks images for unsafe content.

        Args:
            images: Images to check.

        Returns:
            Tuple of images and corresponding NSFW flags.
        """
        safety_checker_input = self._feature_extractor(images, return_tensors="pt").to(
            self.device
        )
        images_np = [np.array(img) for img in images]
        _, has_nsfw_concept = self._safety_checker(
            images=images_np,
            clip_input=safety_checker_input.pixel_values.to(self._dtype),
        )
        return images, has_nsfw_concept

class DirectoryReader:
    def __init__(self, dir: str):
        self.paths = sorted(
            glob.glob(os.path.join(dir, "*")),
            key=lambda x: int(os.path.basename(x).split(".")[0]),
        )
        self.nb_frames = len(self.paths)
        self.idx = 0

        assert self.nb_frames > 0, "no frames found in directory"

        first_img = Image.open(self.paths[0])
        self.height = first_img.height
        self.width = first_img.width

    def get_resolution(self):
        return self.height, self.width

    def reset(self):
        self.idx = 0  # Reset the index counter to 0

    def get_frame(self):
        if self.idx >= self.nb_frames:
            return None

        path = self.paths[self.idx]
        self.idx += 1

        img = Image.open(path)
        transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        return transforms(img) 

class DirectoryWriter:
    def __init__(self, dir: str):
        self.dir = dir
        self.idx = 0

    def open(self):
        return

    def close(self):
        return

    def write_frame(self, frame: torch.Tensor):
        path = f"{self.dir}/{self.idx}.png"
        self.idx += 1

        transforms = v2.Compose([v2.ToPILImage()])
        transforms(frame.squeeze(0)).save(path)