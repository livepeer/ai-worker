from abc import ABC, abstractmethod
from PIL import Image

class Pipeline(ABC):
    """Abstract base class for image processing pipelines.

    Processes frames sequentially and supports dynamic parameter updates.

    Notes:
    - Methods are only called one at a time in a separate process, so no need
      for any locking.
    - Error handling is done by the caller, so the implementation can let
      exceptions propagate for optimal error reporting.
    """

    def __init__(self, **params):
        """Initialize pipeline with optional parameters.

        Args:
            **params: Parameters to initalize the pipeline with.
        """
        pass

    @abstractmethod
    def process_frame(self, frame: Image.Image) -> Image.Image:
        """Process a single frame through the pipeline.

        Called sequentially with each frame from the stream.

        Args:
            frame: Input PIL Image

        Returns:
            Processed PIL Image
        """
        pass

    @abstractmethod
    def update_params(self, **params):
        """Update pipeline parameters.

        Must maintain valid state on success or restore previous state on failure.
        Called sequentially with process_frame so concurrency is not an issue.

        Args:
            **params: Implementation-specific parameters
        """
        pass

    async def stop(self):
        """Stop the pipeline.

        Called once when the pipeline is no longer needed.
        """
        pass
