import base64
import io
import os
from typing import List

from fastapi import UploadFile
from PIL import Image
from pydantic import BaseModel, Field

import numpy as np

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


class chunk(BaseModel):
    """A chunk of text with a timestamp."""

    timestamp: tuple = Field(..., description="The timestamp of the chunk.")
    text: str = Field(..., description="The text of the chunk.")


class TextResponse(BaseModel):
    """Response model for text generation."""

    text: str = Field(..., description="The generated text.")
    chunks: List[chunk] = Field(..., description="The generated text chunks.")


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
    if isinstance(img, np.ndarray):
        img = Image.fromarray((img * 255).astype(np.uint8))
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
