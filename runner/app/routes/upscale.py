import logging
import os
import random
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile
from pydantic import BaseModel

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.util import HTTPError, ImageResponse, http_error, image_to_data_url

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

logger = logging.getLogger(__name__)

class UpscaleParams(BaseModel):
    prompt: str
    image: UploadFile
    model_id: str = ""
    safety_check: bool = True
    seed: int = None
    num_inference_steps: int = 75  # NOTE: Hardcoded due to varying pipeline values.

RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}


# TODO: Make model_id and other None properties optional once Go codegen tool supports
# OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
@router.post("/upscale", response_model=ImageResponse, responses=RESPONSES)
@router.post(
    "/upscale/",
    response_model=ImageResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def upscale(
    params: UpscaleParams,
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    auth_token = os.environ.get("AUTH_TOKEN")
    if auth_token:
        if not token or token.credentials != auth_token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
                content=http_error("Invalid bearer token"),
            )

    if params.model_id != "" and params.model_id != pipeline.model_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{params.model_id}"
            ),
        )

    seed = params.seed or random.randint(0, 2**32 - 1)

    image = Image.open(params.image.file).convert("RGB")

    try:
        images, has_nsfw_concept = pipeline(
            prompt=params.prompt,
            image=image,
            num_inference_steps=params.num_inference_steps,
            safety_check=params.safety_check,
            seed=seed,
        )
    except Exception as e:
        logger.error(f"UpscalePipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=http_error("UpscalePipeline error"),
        )

    seeds = [seed]

    # TODO: Return None once Go codegen tool supports optional properties
    # OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    output_images = [
        {"url": image_to_data_url(img), "seed": sd, "nsfw": nsfw or False}
        for img, sd, nsfw in zip(images, seeds, has_nsfw_concept)
    ]

    return {"images": output_images}
