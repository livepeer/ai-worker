import base64
import io
from typing import List

from PIL import Image
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
