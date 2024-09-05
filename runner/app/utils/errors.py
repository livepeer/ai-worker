"""Custom exceptions used throughout the whole application."""


class InferenceError(Exception):
    """Exception raised for errors during model inference."""

    def __init__(self, message="Error during model execution", original_exception=None):
        """Initialize the exception.

        Args:
            message: The error message.
            original_exception: The original exception that caused the error.
        """
        if original_exception:
            message = f"{message}: {original_exception}"
        super().__init__(message)
        self.original_exception = original_exception


class OutOfMemoryError(Exception):
    """Raised when the system runs out of memory."""

    def __init__(self, message="GPU ran out of memory."):
        self.message = message
        super().__init__(self.message)
