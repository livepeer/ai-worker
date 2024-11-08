from typing import List
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch
import json

LORA_LIMIT = 4  # Max number of LoRas that can be requested at once.
LORA_MAX_LOADED = 12  # Number of LoRas to keep in memory.
LORA_FREE_VRAM_THRESHOLD = 2.0  # VRAM threshold (GB) to start evicting LoRas.


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

class LoraLoader:
    """Utility class to load LoRas and set their weights into a given pipeline.

    Attributes:
        pipeline: Diffusion pipeline on which the LoRas are loaded.
        loras_enabled: Flag to enable or disable LoRas.
    """

    def __init__(self, pipeline: DiffusionPipeline):
        """Initializes the LoraLoader.

        Args:
            pipeline: Diffusion pipeline to load LoRas into.
        """
        self.pipeline = pipeline
        self.loras_enabled = False

    def _get_loaded_loras(self) -> List[str]:
        """Returns the names of the loaded LoRas.

        Returns:
            List of loaded LoRa names.
        """
        loaded_loras_dict = self.pipeline.get_list_adapters()
        seen = set()
        return [
            lora
            for loras in loaded_loras_dict.values()
            for lora in loras
            if lora not in seen and not seen.add(lora)
        ]

    def _evict_loras_if_needed(self, request_loras: dict) -> None:
        """Evict the oldest unused LoRa until free memory is above the threshold or the
        number of loaded LoRas is below the maximum allowed.

        Args:
            request_loras: list of requested LoRas.
        """
        while True:
            free_memory_gb = (
                torch.cuda.mem_get_info(device=self.pipeline.device)[0] / 1024**3
            )
            loaded_loras = self._get_loaded_loras()
            memory_limit_reached = free_memory_gb < LORA_FREE_VRAM_THRESHOLD

            # Break if memory is sufficient, LoRas within limit, or no LoRas to evict.
            if (
                not memory_limit_reached
                and len(loaded_loras) < LORA_MAX_LOADED
                or not any(lora not in request_loras for lora in loaded_loras)
            ):
                break

            # Evict the oldest unused LoRa.
            for lora in loaded_loras:
                if lora not in request_loras:
                    self.pipeline.delete_adapters(lora)
                    break
        if memory_limit_reached:
            torch.cuda.empty_cache()

    def load_loras(self, loras_json: str) -> None:
        """Loads LoRas and sets their weights into the pipeline managed by this
        LoraLoader.

        Args:
            loras_json: A JSON string containing key-value pairs, where the key is the
                repository to load LoRas from and the value is the strength (a float
                with a minimum value of 0.0) to assign to the LoRa.

        Raises:
            LoraLoadingError: If an error occurs during LoRa loading.
        """
        try:
            lora_dict = json.loads(loras_json)
        except json.JSONDecodeError:
            error_message = f"Unable to parse '{loras_json}' as JSON."
            logger.warning(error_message)
            raise LoraLoadingError(error_message)

        # Parse Lora strengths and check for invalid values.
        invalid_loras = {
            adapter: val
            for adapter, val in lora_dict.items()
            if not is_numeric(val) or float(val) < 0.0
        }
        if invalid_loras:
            error_message = (
                "All strengths must be numbers greater than or equal to 0.0."
            )
            logger.warning(error_message)
            raise LoraLoadingError(error_message)
        lora_dict = {adapter: float(val) for adapter, val in lora_dict.items()}

        # Disable LoRas if none are provided.
        if not lora_dict:
            self.disable_loras()
            return

        # Limit the number of active loras to prevent pipeline slowdown.
        if len(lora_dict) > LORA_LIMIT:
            raise LoraLoadingError(f"Too many LoRas provided. Maximum is {LORA_LIMIT}.")

        # Re-enable LoRas if they were disabled.
        self.enable_loras()

        # Load new LoRa adapters.
        loaded_loras = self._get_loaded_loras()
        try:
            for adapter in lora_dict.keys():
                # Load new Lora weights and evict the oldest unused Lora if necessary.
                if adapter not in loaded_loras:
                    self.pipeline.load_lora_weights(adapter, adapter_name=adapter)
                    self._evict_loras_if_needed(list(lora_dict.keys()))
        except Exception as e:
            # Delete failed adapter and log the error.
            self.pipeline.delete_adapters(adapter)
            torch.cuda.empty_cache()
            if "not found in the base model" in str(e):
                error_message = (
                    "LoRa incompatible with base model: "
                    f"'{self.pipeline.name_or_path}'"
                )
            elif getattr(e, "server_message", "") == "Repository not found":
                error_message = f"LoRa repository '{adapter}' not found"
            else:
                error_message = f"Unable to load LoRas for adapter '{adapter}'"
                logger.exception(e)
            raise LoraLoadingError(error_message)

        # Set unused LoRas strengths to 0.0.
        for lora in loaded_loras:
            if lora not in lora_dict:
                lora_dict[lora] = 0.0

        # Set the lora adapter strengths.
        self.pipeline.set_adapters(*map(list, zip(*lora_dict.items())))

    def disable_loras(self) -> None:
        """Disables all LoRas in the pipeline."""
        if self.loras_enabled:
            self.pipeline.disable_lora()
            self.loras_enabled = False

    def enable_loras(self) -> None:
        """Enables all LoRas in the pipeline."""
        if not self.loras_enabled:
            self.pipeline.enable_lora()
            self.loras_enabled = True
