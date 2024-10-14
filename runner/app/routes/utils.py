import base64
import io
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from fastapi import UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field


class Media(BaseModel):
    """A media object containing information about the generated media."""

    url: str = Field(..., description="The URL where the media can be accessed.")
    seed: int = Field(..., description="The seed used to generate the media.")
    # TODO: Make nsfw property optional once Go codegen tool supports
    # OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    nsfw: bool = Field(..., description="Whether the media was flagged as NSFW.")


class ImageResponse(BaseModel):
    """Response model for image generation."""

    images: List[Media] = Field(..., description="The generated images.")


class VideoResponse(BaseModel):
    """Response model for video generation."""

    frames: List[List[Media]] = Field(..., description="The generated video frames.")


class MasksResponse(BaseModel):
    """Response model for object segmentation."""

    masks: str = Field(..., description="The generated masks.")
    scores: str = Field(
        ..., description="The model's confidence scores for each generated mask."
    )
    logits: str = Field(
        ..., description="The raw, unnormalized predictions (logits) for the masks."
    )


class chunk(BaseModel):
    """A chunk of text with a timestamp."""

    timestamp: tuple = Field(..., description="The timestamp of the chunk.")
    text: str = Field(..., description="The text of the chunk.")


class TextResponse(BaseModel):
    """Response model for text generation."""

    text: str = Field(..., description="The generated text.")
    chunks: List[chunk] = Field(..., description="The generated text chunks.")


class LLMResponse(BaseModel):
    response: str
    tokens_used: int


class APIError(BaseModel):
    """API error response model."""

    msg: str = Field(..., description="The error message.")


class HTTPError(BaseModel):
    """HTTP error response model."""

    detail: APIError = Field(..., description="Detailed error information.")


def http_error(msg: str) -> HTTPError:
    """Create an HTTP error response with the specified message.

    Args:
        msg: The error message.

    Returns:
        The HTTP error response.
    """
    return {"detail": {"msg": msg}}


def image_to_base64(img: Image, format: str = "png") -> str:
    """Convert an image to a base64 string.

    Args:
        img: The image to convert.
        format: The image format to use. Defaults to "png".

    Returns:
        The base64-encoded image.
    """
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def image_to_data_url(img: Image, format: str = "png") -> str:
    """Convert an image to a data URL.

    Args:
        img: The image to convert.
        format: The image format to use. Defaults to "png".

    Returns:
        The data URL for the image.
    """
    return "data:image/png;base64," + image_to_base64(img, format=format)


def file_exceeds_max_size(
    input_file: UploadFile, max_size: int = 10 * 1024 * 1024
) -> bool:
    """Checks if the uploaded file exceeds the specified maximum size.

    Args:
        input_file: The uploaded file to check.
        max_size: The maximum allowed file size in bytes. Defaults to 10 MB.

    Returns:
        True if the file exceeds the maximum size, False otherwise.
    """
    try:
        if input_file.file:
            # Get size by moving the cursor to the end of the file and back.
            input_file.file.seek(0, os.SEEK_END)
            file_size = input_file.file.tell()
            input_file.file.seek(0)
            return file_size > max_size
    except Exception as e:
        print(f"Error checking file size: {e}")
    return False


def json_str_to_np_array(
    data: Optional[str], var_name: Optional[str] = None
) -> Optional[np.ndarray]:
    """Converts a JSON string to a NumPy array.

    Args:
        data: The JSON string to convert.
        var_name: The name of the variable being converted. Used in error messages.

    Returns:
        The NumPy array if the conversion is successful, None otherwise.

    Raises:
        ValueError: If an error occurs during JSON parsing.
    """
    if data:
        try:
            array = np.array(json.loads(data))
            return array
        except json.JSONDecodeError as e:
            error_message = "Error parsing JSON"
            if var_name:
                error_message += f" for {var_name}"
            error_message += f": {e}"
            raise ValueError(error_message)
    return None


# Global error handling configuration.
ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    # Specific error types.
    "LoraLoadingError": (None, status.HTTP_400_BAD_REQUEST),
    "InferenceError": (None, status.HTTP_400_BAD_REQUEST),
    "ValueError": ("Pipeline error.", status.HTTP_400_BAD_REQUEST),
    "OutOfMemoryError": ("GPU out of memory.", status.HTTP_500_INTERNAL_SERVER_ERROR),
    # General error patterns.
    "out of memory": ("Out of memory.", status.HTTP_500_INTERNAL_SERVER_ERROR),
    "CUDA out of memory": ("GPU out of memory.", status.HTTP_500_INTERNAL_SERVER_ERROR),
}


def handle_pipeline_exception(
    e: object,
    default_error_message: Union[str, Dict[str, object]] = "Pipeline error.",
    default_status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    custom_error_config: Dict[str, Tuple[str, int]] = None,
) -> JSONResponse:
    """Handles pipeline exceptions by returning a JSON response with the appropriate
    error message and status code.

    Args:
        e (object): The exception to handle. Can be any type of object.
        default_error_message (Union[str, Dict[str, Any]]): The default error message
            or content dictionary. Default will be used if no specific error type is
            matched.
        default_status_code (int): The default status code to use if no specific error
            type is matched. Defaults to HTTP_500_INTERNAL_SERVER_ERROR.
        custom_error_config (Dict[str, Tuple[str, int]]): Custom error configuration
            to override the application error configuration.

    Returns:
        JSONResponse: The JSON response with appropriate status code and error message.
    """
    error_config = ERROR_CONFIG.copy()

    # Update error_config with custom_error_config if provided.
    if custom_error_config:
        error_config.update(custom_error_config)

    error_message = default_error_message
    status_code = default_status_code

    error_type = type(e).__name__
    if error_type in error_config:
        message, status_code = error_config[error_type]
        error_message = str(e) if message is None or message == "" else message
    else:
        for error_pattern, (message, code) in error_config.items():
            if error_pattern.lower() in str(e).lower():
                error_message = str(e) if message is None or message == "" else message
                status_code = code
                break

    if isinstance(error_message, str):
        content = http_error(error_message)
    else:
        content = error_message

    return JSONResponse(
        status_code=status_code,
        content=content,
    )
