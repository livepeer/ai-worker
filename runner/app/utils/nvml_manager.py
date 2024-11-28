"""This module manages NVML (NVIDIA Management Library) initialization and shutdown,
ensuring efficient resource management and improved performance for GPU operations.
"""

import pynvml
import logging
import atexit

logger = logging.getLogger(__name__)


class NVMLManager:
    """A class to manage NVML initialization and shutdown.

    Attributes:
        initialized (bool): Whether NVML is initialized.
    """

    def __init__(self):
        self._initialized = False
        atexit.register(self.shutdown)

    def initialize(self) -> None:
        """Initialize NVML."""
        if self._initialized:
            logger.warning("NVML is already initialized.")
            return

        try:
            pynvml.nvmlInit()
            self._initialized = True
            logger.info("NVML initialized successfully.")
        except (ImportError, pynvml.NVMLError) as e:
            logger.error(f"Failed to initialize NVML: {e}")

    def shutdown(self) -> None:
        """Shutdown NVML."""
        if not self._initialized:
            logger.warning("NVML is not initialized.")
            return

        try:
            pynvml.nvmlShutdown()
            self._initialized = False
            logger.info("NVML shutdown successfully.")
        except pynvml.NVMLError as e:
            logger.error(f"Failed to shutdown NVML: {e}")

    @property
    def initialized(self):
        """Return whether NVML is initialized."""
        return self._initialized

    @property
    def pynvml(self):
        """Ensure NVML is initialized and return the pynvml module."""
        if not self._initialized:
            self.initialize()
        if self._initialized:
            return pynvml
        else:
            raise RuntimeError("NVML is not initialized")


nvml_manager = NVMLManager()
