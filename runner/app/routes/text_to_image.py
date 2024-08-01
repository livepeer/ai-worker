import logging
import os
import random
from typing import Annotated

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.util import HTTPError, ImageResponse, http_error, image_to_data_url

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

router = APIRouter()

logger = logging.getLogger(__name__)


class TextToImageParams(BaseModel):
    # TODO: Make model_id and other None properties optional once Go codegen tool
    # supports OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    model_id: str = ""
    prompt: str
    height: int = None
    width: int = None
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    safety_check: bool = True
    seed: int = None
    num_inference_steps: int = 50  # NOTE: Hardcoded due to varying pipeline values.
    num_images_per_prompt: int = 1


RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}


@router.post("/text-to-image", response_model=ImageResponse, responses=RESPONSES)
@router.post(
    "/text-to-image/",
    response_model=ImageResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def text_to_image(
    params: TextToImageParams,
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

    seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)
    seeds = [seed + i for i in range(params.num_images_per_prompt)]

    # TODO: Process one image at a time to avoid CUDA OEM errors. Can be removed again
    # once LIV-243 and LIV-379 are resolved.
    images = []
    has_nsfw_concept = []
    num_images_per_prompt = 1
    for seed in seeds:
        try:
            seed = seed
            kwargs = {k: v for k, v in params.model_dump().items() if k != "model_id"}
            imgs, nsfw_check = pipeline(**kwargs)
            images.extend(imgs)
            has_nsfw_concept.extend(nsfw_check)
        except Exception as e:
            logger.error(f"TextToImagePipeline error: {e}")
            logger.exception(e)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=http_error("TextToImagePipeline error"),
            )

    # TODO: Return None once Go codegen tool supports optional properties
    # OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    output_images = [
        {"url": image_to_data_url(img), "seed": sd, "nsfw": nsfw or False}
        for img, sd, nsfw in zip(images, seeds, has_nsfw_concept)
    ]

    return {"images": output_images}
