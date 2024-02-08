import io
import PIL
import base64
from pydantic import BaseModel
from typing import List


class Media(BaseModel):
    url: str
    seed: int


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


def image_to_base64(img: PIL.Image, format: str = "png") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def image_to_data_url(img: PIL.Image, format: str = "png") -> str:
    return "data:image/png;base64," + image_to_base64(img, format=format)
