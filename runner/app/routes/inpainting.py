# runner/app/routes/inpainting.py

import logging
import os
import random
from typing import Annotated, Dict, Tuple, Union

import torch
from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import (
    HTTPError,
    ImageResponse,
    http_error,
    image_to_data_url,
    handle_pipeline_exception,
)
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()
logger = logging.getLogger(__name__)

# Pipeline specific error handling configuration
PIPELINE_ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    "OutOfMemoryError": (
        "Out of memory error. Try reducing input image resolution.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    ),
    "InferenceError": (None, status.HTTP_400_BAD_REQUEST),
    "ValueError": (None, status.HTTP_400_BAD_REQUEST),
}

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
    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}

@router.post(
    "/inpainting",
    response_model=ImageResponse,
    responses=RESPONSES,
    description="Apply inpainting transformations to a provided image using a mask.",
    operation_id="genInpainting",
    summary="Inpainting",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "inpainting"},
)
@router.post(
    "/inpainting/",
    response_model=ImageResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def inpainting(
    prompt: Annotated[
        str,
        Form(description="Text prompt(s) to guide image generation."),
    ],
    image: Annotated[
        UploadFile,
        File(description="Original image to be modified."),
    ],
    mask_image: Annotated[
        UploadFile,
        File(description="Mask image indicating areas to be inpainted."),
    ],
    model_id: Annotated[
        str,
        Form(description="Hugging Face model ID used for inpainting."),
    ] = "",
    loras: Annotated[
        str,
        Form(description="LoRA models and weights for image generation."),
    ] = "",
    negative_prompt: Annotated[
        str,
        Form(description="Text prompt(s) to guide what to exclude."),
    ] = "",
    strength: Annotated[
        float,
        Form(description="Strength of the inpainting effect (0 to 1)."),
    ] = 1.0,
    guidance_scale: Annotated[
        float,
        Form(description="How closely to follow the prompt."),
    ] = 7.5,
    safety_check: Annotated[
        bool,
        Form(description="Check for NSFW content."),
    ] = True,
    seed: Annotated[
        int, 
        Form(description="Random seed for generation.")
    ] = None,
    num_inference_steps: Annotated[
        int,
        Form(description="Number of denoising steps."),
    ] = 50,
    num_images_per_prompt: Annotated[
        int,
        Form(description="Number of images to generate per prompt."),
    ] = 1,
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
                f"pipeline configured with {pipeline.model_id} but called with {model_id}."
            ),
        )

    if not 0 <= strength <= 1:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error("Strength must be between 0 and 1."),
        )

    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    seeds = [seed + i for i in range(num_images_per_prompt)]

    try:
        input_image = Image.open(image.file).convert("RGB")
        mask = Image.open(mask_image.file).convert("RGB")
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(f"Error loading images: {str(e)}"),
        )

    images = []
    has_nsfw_concept = []
    for seed in seeds:
        try:
            imgs, nsfw_checks = pipeline(
                prompt=prompt,
                image=input_image,
                mask_image=mask,
                strength=strength,
                loras=loras,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                safety_check=safety_check,
                seed=seed,
                num_images_per_prompt=1,
                num_inference_steps=num_inference_steps,
            )
        except Exception as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            logger.error(f"Inpainting pipeline error: {e}")
            return handle_pipeline_exception(
                e,
                default_error_message="Inpainting pipeline error.",
                custom_error_config=PIPELINE_ERROR_CONFIG,
            )
        images.extend(imgs)
        has_nsfw_concept.extend(nsfw_checks)

    output_images = [
        {"url": image_to_data_url(img), "seed": sd, "nsfw": nsfw or False}
        for img, sd, nsfw in zip(images, seeds, has_nsfw_concept)
    ]

    return {"images": output_images}