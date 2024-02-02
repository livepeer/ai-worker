from fastapi import Depends, APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.pipelines import ImageToImagePipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, ImageResponse, HTTPError, http_error
import PIL
from typing import Annotated
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

responses = {400: {"model": HTTPError}, 500: {"model": HTTPError}}


# TODO: Make model_id optional once Go codegen tool supports OAPI 3.1
# https://github.com/deepmap/oapi-codegen/issues/373
@router.post("/image-to-image", response_model=ImageResponse, responses=responses)
@router.post(
    "/image-to-image/",
    response_model=ImageResponse,
    responses=responses,
    include_in_schema=False,
)
async def image_to_image(
    prompt: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
    model_id: Annotated[str, Form()] = "",
    strength: Annotated[float, Form()] = 0.8,
    guidance_scale: Annotated[float, Form()] = 7.5,
    negative_prompt: Annotated[str, Form()] = "",
    seed: Annotated[int, Form()] = None,
    pipeline: ImageToImagePipeline = Depends(get_pipeline),
):
    if model_id != "" and model_id != pipeline.model_id:
        return JSONResponse(
            status_code=400,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with {model_id}"
            ),
        )

    try:
        images = pipeline(
            prompt,
            PIL.Image.open(image.file).convert("RGB"),
            strength=strength,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            seed=seed,
        )
    except Exception as e:
        logger.error(f"ImageToImagePipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=500, content=http_error("ImageToImagePipeline error")
        )

    output_images = []
    for img in images:
        output_images.append({"url": image_to_data_url(img)})

    return {"images": output_images}
