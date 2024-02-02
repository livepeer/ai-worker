from pydantic import BaseModel
from fastapi import Depends, APIRouter
from fastapi.responses import JSONResponse
from app.pipelines import TextToImagePipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, ImageResponse, HTTPError, http_error
import logging

router = APIRouter()

logger = logging.getLogger(__name__)


class TextToImageParams(BaseModel):
    # TODO: Make model_id optional once Go codegen tool supports OAPI 3.1
    # https://github.com/deepmap/oapi-codegen/issues/373
    model_id: str = ""
    prompt: str
    height: int = None
    width: int = None
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    seed: int = None


responses = {400: {"model": HTTPError}, 500: {"model": HTTPError}}


@router.post("/text-to-image", response_model=ImageResponse, responses=responses)
@router.post("/text-to-image/", response_model=ImageResponse, include_in_schema=False)
async def text_to_image(
    params: TextToImageParams, pipeline: TextToImagePipeline = Depends(get_pipeline)
):
    if params.model_id != "" and params.model_id != pipeline.model_id:
        return JSONResponse(
            status_code=400,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with {params.model_id}"
            ),
        )

    try:
        images = pipeline(**params.model_dump())
    except Exception as e:
        logger.error(f"TextToImagePipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=500, content=http_error("TextToImagePipeline error")
        )

    output_images = []
    for img in images:
        output_images.append({"url": image_to_data_url(img)})

    return {"images": output_images}
