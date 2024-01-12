from pydantic import BaseModel
from fastapi import Depends, APIRouter
from app.pipelines import TextToImagePipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, Media
from typing import List

router = APIRouter()


class TextToImageParams(BaseModel):
    prompt: str


class TextToImageResponse(BaseModel):
    images: List[Media]


@router.post("/text-to-image", response_model=TextToImageResponse)
def text_to_image(
    params: TextToImageParams, pipeline: TextToImagePipeline = Depends(get_pipeline)
):
    images = pipeline(params.prompt)

    output_images = []
    for img in images:
        output_images.append({"url": image_to_data_url(img)})

    return {"images": output_images}
