from pydantic import BaseModel, ConfigDict
from fastapi import Depends, APIRouter
from app.pipelines import TextToImagePipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, ImageResponse
from typing import Optional

router = APIRouter()


class TextToImageParams(BaseModel):
    # TODO: Make model_id optional once Go codegen tool supports OAPI 3.1
    # https://github.com/deepmap/oapi-codegen/issues/373
    model_id: str = ""
    prompt: str


@router.post("/text-to-image", response_model=ImageResponse)
@router.post("/text-to-image/", response_model=ImageResponse, include_in_schema=False)
async def text_to_image(
    params: TextToImageParams, pipeline: TextToImagePipeline = Depends(get_pipeline)
):
    if params.model_id != "" and params.model_id != pipeline.model_id:
        raise Exception(
            f"pipeline configured with {pipeline.model_id} but called with {params.model_id}"
        )

    images = pipeline(params.prompt)

    output_images = []
    for img in images:
        output_images.append({"url": image_to_data_url(img)})

    return {"images": output_images}
