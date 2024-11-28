"""
This module manages NVML (NVIDIA Management Library) initialization and shutdown,
ensuring efficient resource management and improved performance for GPU operations.
"""
import pynvml
import logging
import atexit

logger = logging.getLogger(__name__)

class NVMLManager:
    """A class to manage NVML initialization and shutdown."""
    def __init__(self):
        self._initialized = False
        atexit.register(self.shutdown)

    def initialize(self):
        """Initialize NVML."""
        if not self._initialized:
            try:
                pynvml.nvmlInit()
                self._initialized = True
                logger.info("NVML initialized successfully.")
            except pynvml.NVMLError as e:
                logger.error(f"Failed to initialize NVML: {e}")

    def shutdown(self):
        """Shutdown NVML."""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
                self._initialized = False
                logger.info("NVML shutdown successfully.")
            except pynvml.NVMLError as e:
                logger.error(f"Failed to shutdown NVML: {e}")

nvml_manager = NVMLManager()
