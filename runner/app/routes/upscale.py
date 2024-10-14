import logging
import os
import random
from typing import Annotated

import torch
from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import HTTPError, ImageResponse, http_error, image_to_data_url
from app.utils.errors import InferenceError
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

logger = logging.getLogger(__name__)


def handle_pipeline_error(e: Exception) -> JSONResponse:
    """Handles exceptions raised during upscale pipeline processing.

    Args:
        e: The exception raised during upscale processing.

    Returns:
        A JSONResponse with the appropriate error message and status code.
    """
    if isinstance(e, torch.cuda.OutOfMemoryError):
        status_code = status.HTTP_400_BAD_REQUEST
        error_message = "Out of memory error. Try reducing input image resolution."
        torch.cuda.empty_cache()
    elif isinstance(e, InferenceError):
        status_code = status.HTTP_400_BAD_REQUEST
        error_message = str(e)
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_message = "Upscale pipeline error."

    return JSONResponse(
        status_code=status_code,
        content=http_error(error_message),
    )


RESPONSES = {
    status.HTTP_200_OK: {
        "content": {
            "application/json": {
                "schema": {
                    "x-speakeasy-name-override": "data",
                }
            }
        },
    },
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}


# TODO: Make model_id and other None properties optional once Go codegen tool supports
# OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
@router.post(
    "/upscale",
    response_model=ImageResponse,
    responses=RESPONSES,
    description="Upscale an image by increasing its resolution.",
    operation_id="genUpscale",
    summary="Upscale",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "upscale"},
)
@router.post(
    "/upscale/",
    response_model=ImageResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def upscale(
    prompt: Annotated[
        str,
        Form(description="Text prompt(s) to guide upscaled image generation."),
    ],
    image: Annotated[
        UploadFile,
        File(description="Uploaded image to modify with the pipeline."),
    ],
    model_id: Annotated[
        str,
        Form(description="Hugging Face model ID used for upscaled image generation."),
    ] = "",
    safety_check: Annotated[
        bool,
        Form(
            description=(
                "Perform a safety check to estimate if generated images could be "
                "offensive or harmful."
            )
        ),
    ] = True,
    seed: Annotated[int, Form(description="Seed for random number generation.")] = None,
    num_inference_steps: Annotated[
        int,
        Form(
            description=(
                "Number of denoising steps. More steps usually lead to higher quality "
                "images but slower inference. Modulated by strength."
            )
        ),
    ] = 75,  # NOTE: Hardcoded due to varying pipeline values.
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    auth_token = os.environ.get("AUTH_TOKEN")
    if auth_token:
        if not token or token.credentials != auth_token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
                content=http_error("Invalid bearer token."),
            )

    if model_id != "" and model_id != pipeline.model_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{model_id}."
            ),
        )

    seed = seed or random.randint(0, 2**32 - 1)

    image = Image.open(image.file).convert("RGB")

    try:
        images, has_nsfw_concept = pipeline(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            safety_check=safety_check,
            seed=seed,
        )
    except Exception as e:
        logger.error(f"TextToImage pipeline error: {e}")
        return handle_pipeline_error(e)

    seeds = [seed]

    # TODO: Return None once Go codegen tool supports optional properties
    # OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    output_images = [
        {"url": image_to_data_url(img), "seed": sd, "nsfw": nsfw or False}
        for img, sd, nsfw in zip(images, seeds, has_nsfw_concept)
    ]

    return {"images": output_images}
