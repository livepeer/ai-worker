import base64
import io
import os
from typing import List

from PIL import Image
from fastapi import HTTPException, UploadFile
from pydantic import BaseModel


class Media(BaseModel):
    url: str
    seed: int
    # TODO: Make nsfw property optional once Go codegen tool supports
    # OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    nsfw: bool

class chunk(BaseModel):
    timestamp: tuple
    text: str

class TextResponse(BaseModel):
    text: str
    chunks: List[chunk]


class ImageResponse(BaseModel):
    images: List[Media]



class VideoResponse(BaseModel):
    frames: List[List[Media]]


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

def verify_file_size(file: UploadFile, max_size: int = 10 * 1024 * 1024):  # 10 MB limit
    """
    Verifies the size of the uploaded file.
    Raises an HTTPException if the file exceeds the specified max_size.
    """
    if file.file:
        # Move the cursor to the end of the file to get its size
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        # Move the cursor back to the beginning of the file for subsequent operations
        file.file.seek(0)
        if file_size > max_size:
            print("File size exceeds limit")
            raise HTTPException(status_code=413, detail="File size exceeds limit")
    return file
