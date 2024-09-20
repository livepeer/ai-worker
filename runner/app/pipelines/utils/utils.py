"""This module contains several utility functions."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from PIL import Image
from torch import dtype as TorchDtype
from transformers import CLIPImageProcessor

logger = logging.getLogger(__name__)


class LoraLoadingError(Exception):
    """Exception raised for errors during LoRa loading."""

    def __init__(self, message="Error loading LoRas", original_exception=None):
        """Initialize the exception.
        Args:
            message: The error message.
            original_exception: The original exception that caused the error.
        """
        if original_exception:
            message = f"{message}: {original_exception}"
        super().__init__(message)
        self.original_exception = original_exception


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


class LoraLoader:
    """Utility class to load LoRas and set their weights into a given pipeline.

    Attributes:
        pipeline: Diffusion pipeline on which the LoRas are loaded.
    """

    def __init__(self, pipeline: DiffusionPipeline):
        """Initializes the LoraLoader.

        Args:
            pipeline: Diffusion pipeline to load LoRas into.
        """
        self.pipeline = pipeline
        self._loaded_loras = set()

    def load_loras(self, loras_json: str) -> None:
        """Loads LoRas and sets their weights into the pipeline managed by this
        LoraLoader.

        Args:
            loras_json: A JSON string containing key-value pairs, where the key is the
                repository to load LoRas from and the value is the strength (a float
                with a minimum value of 0.0) to assign to the LoRa.
        """
        try:
            lora_dict = json.loads(loras_json)
        except json.JSONDecodeError:
            error_message = f"Unable to parse '{loras_json}' as JSON."
            logger.warning(error_message)
            raise LoraLoadingError(error_message)

        invalid_loras = {
            adapter: val
            for adapter, val in lora_dict.items()
            if not isinstance(val, (int, float)) or val < 0.0
        }
        if invalid_loras:
            error_message = (
                "All strengths must be numbers greater than or equal to 0.0."
            )
            logger.warning(error_message)
            raise LoraLoadingError(error_message)

        # Unload LoRas that are no longer needed
        new_loras = set(lora_dict.keys())
        loaded_loras = set(self.pipeline.get_active_adapters())
        loras_to_unload = loaded_loras - new_loras
        for lora in loras_to_unload:
            self.pipeline.unload_lora_weights()
            # self.pipeline.delete_adapters(lora)

        # Load and set weights for each LoRa.
        for adapter in lora_dict.keys():
            # Load the LoRa weights only if not already loaded.
            if adapter not in self._loaded_loras:
                try:
                    self.pipeline.load_lora_weights(adapter, adapter_name=adapter)
                    self._loaded_loras.add(adapter)
                except Exception:
                    error_message = (
                        f"Unable to load LoRas for adapter '{adapter}'"
                    )
                    logger.warning(error_message)
                    raise LoraLoadingError(error_message)

        # Set the adapters and their strengths.
        adapters, strengths = zip(*lora_dict.items())
        self.pipeline.set_adapters(list(adapters), list(strengths))
