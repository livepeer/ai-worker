"""This module contains several utility functions."""

import logging
import os
import json
import numpy as np
from torch import dtype as TorchDtype
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torch import dtype as TorchDtype
from transformers import CLIPImageProcessor

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

def load_loras(pipeline: any, requested_lora: str, loaded_loras: list) -> list:
    """Loads LoRas and sets their weights into the given pipeline.

    Args:
        pipeline: Diffusion pipeline, usually available under self.ldm.
        loras: JSON string with key-value pairs, where the key is the repository to load LoRas from and the value is the strength (float with a minimum value of 0.0) to assign to the LoRa.
    """
    # No LoRas to load
    if requested_lora == "" or requested_lora == None:
        return
    # Parse LoRas param as JSON to extract key-value pairs
    # Build a list of adapter names and their requested strength
    adapters = []
    strengths = []
    if len(loaded_loras) > 0: 
        pipeline.unload_lora_weights()
        
    try:
        lora = json.loads(requested_lora)
    except Exception as e:
        logger.warning(
            f"Unable to parse '{requested_lora}' as JSON. Continuing inference without loading this LoRa"
        )
        
    for adapter, val in lora.items():
        if adapter in loaded_loras:
            pipeline.unload_lora_weights()

        # Sanity check: strength should be a number with a minimum value of 0.0
        try:
            strength = float(val)
        except ValueError:
            logger.warning(
                "Skipping requested LoRa " + adapter + ", as it's requested strength (" + val + ") is not a number"
            )
            continue
        if strength < 0.0:
            logger.warning(
                "Clipping strength of LoRa " + adapter + " to 0.0, as it's requested strength (" + val + ") is negative"
            )
            strength = 0.0
        try:
            # TODO: If we decide to keep LoRas loaded (and only set their weight to 0), make sure that reloading them causes no performance hit or other issues
            pipeline.load_lora_weights(adapter, adapter_name=adapter)
        except Exception as e:
            logger.warning(
                "Unable to load LoRas for adapter '" + adapter + "' (" + type(e).__name__ + ")"
            )
            continue
        # Remember adapter name and their associated strength
        adapters.append(adapter)
        strengths.append(strength)
    # Set weights for all loaded adapters
    if len(adapters) > 0:
        pipeline.set_adapters(adapters, strengths)
    return adapters