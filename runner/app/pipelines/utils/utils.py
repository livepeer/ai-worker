"""This module contains several utility functions."""

import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
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


def get_temp_file(prefix: str, extension: str) -> str:
    """Generates a temporary file path with the specified prefix and extension.

    Args:
        prefix: The prefix for the temporary file.
        extension: The extension for the temporary file.

    Returns:
        The path to a non-existing temporary file with the specified prefix and extension.
    """
    if not extension.startswith("."):
        extension = "." + extension
    filename = f"{prefix}{uuid.uuid4()}{extension}"
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    while os.path.exists(temp_path):
        filename = f"{prefix}{uuid.uuid4()}{extension}"
        temp_path = os.path.join(tempfile.gettempdir(), filename)
    return temp_path


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
