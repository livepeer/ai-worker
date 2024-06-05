from fastapi import Depends, APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.pipelines.base import Pipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, ImageResponse, HTTPError, http_error
from PIL import Image
from typing import Annotated
import logging
import random
import os

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

logger = logging.getLogger(__name__)

responses = {400: {"model": HTTPError}, 500: {"model": HTTPError}}


# TODO: Make model_id and other None properties optional once Go codegen tool supports
# OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
@router.post("/upscale", response_model=ImageResponse, responses=responses)
@router.post(
    "/upscale/",
    response_model=ImageResponse,
    responses=responses,
    include_in_schema=False,
)
async def upscale(
    prompt: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
    model_id: Annotated[str, Form()] = "",
    safety_check: Annotated[bool, Form()] = True,
    seed: Annotated[int, Form()] = None,
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    auth_token = os.environ.get("AUTH_TOKEN")
    if auth_token:
        if not token or token.credentials != auth_token:
            return JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
                content=http_error("Invalid bearer token"),
            )

    if model_id != "" and model_id != pipeline.model_id:
        return JSONResponse(
            status_code=400,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{model_id}"
            ),
        )

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    image = Image.open(image.file).convert("RGB")

    try:
        images  , has_nsfw_concept = pipeline(
            prompt=prompt,
            image=image,
            safety_check=safety_check,
            seed=seed,
        )
    except Exception as e:
        logger.error(f"UpscalePipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=500, content=http_error("UpscalePipeline error")
        )

    seeds = seed if isinstance(seed, list) else [seed]

    output_images = []
    for img, sd, is_nsfw in zip(images, seeds, has_nsfw_concept):
        # TODO: Return None once Go codegen tool supports optional properties
        # OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
        is_nsfw = is_nsfw or False
        output_images.append(
            {"url": image_to_data_url(img), "seed": sd, "nsfw": is_nsfw}
        )

    return {"images": output_images}