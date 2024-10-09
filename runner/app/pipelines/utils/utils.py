"""This module contains several utility functions."""

import os
import re
import cv2
import glob
import torch
import logging
import tempfile
import numpy as np

from PIL import Image
from pathlib import Path
from typing import Dict, Optional
from torch import dtype as TorchDtype
from torchvision.transforms import v2
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

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
) -> Dict[str, str]:
    """Splits an input prompt into prompts, including the main prompt, with customizable
    key naming.

    Args:
        input_prompt (str): The input prompt string to be split.
        separator (str): The character used to split the input prompt. Defaults to '|'.
        key_prefix (str): Prefix for keys in the returned dictionary for all prompts,
            including the main prompt. Defaults to 'prompt'.
        max_splits (int): Maximum number of splits to perform. Defaults to -1 (no
            limit).

    Returns:
        Dict[str, str]: A dictionary of all prompts, including the main prompt.
    """
    prompts = [
        prompt.strip()
        for prompt in input_prompt.split(separator, max_splits)
        if prompt.strip()
    ]
    if not prompts:
        return {}

    start_index = max(1, len(prompts) - max_splits) if max_splits >= 0 else 1

    prompt_dict = {f"{key_prefix}": prompts[0]}
    prompt_dict.update(
        {
            f"{key_prefix}_{i+1}": prompt
            for i, prompt in enumerate(prompts[1:], start=start_index)
        }
    )

    return prompt_dict

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
            temp_file.write(video_data)
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
        self._feature_extractor = CLIPImageProcessor.from_pretrained(
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


def natural_sort_key(s):
    """
    Sort in a natural order, separating strings into a list of strings and integers.
    This handles leading zeros and case insensitivity.
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'([0-9]+)', os.path.basename(s))
    ]

class DirectoryReader:
    def __init__(self, dir: str):
        self.paths = sorted(
            glob.glob(os.path.join(dir, "*")),
            key=natural_sort_key
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
        try:
            img = Image.open(path)
            transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
            frame = transforms(img)
            self.idx += 1
            return frame
        except Exception as e:
            logger.error(f"Error reading frame {self.idx}: {e}")
            return None

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
