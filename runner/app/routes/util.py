import base64
import io
import os
from typing import List

from fastapi import UploadFile
from PIL import Image
from pydantic import BaseModel


class Media(BaseModel):
    url: str
    seed: int
    # TODO: Make nsfw property optional once Go codegen tool supports
    # OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    nsfw: bool


class ImageResponse(BaseModel):
    images: List[Media]


class VideoResponse(BaseModel):
    frames: List[List[Media]]


class chunk(BaseModel):
    timestamp: tuple
    text: str


class TextResponse(BaseModel):
    text: str
    chunks: List[chunk]


class LlmResponse(BaseModel):
    response: str
    tokens_used: int


class APIError(BaseModel):
    msg: str


class HTTPError(BaseModel):
    detail: APIError


def http_error(msg: str) -> HTTPError:
    return {"detail": {"msg": msg}}


def image_to_base64(img: Image, format: str = "png") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def image_to_data_url(img: Image, format: str = "png") -> str:
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
