import logging
import os
from typing import Annotated

import torch
from fastapi import APIRouter, Depends, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.util import HTTPError, ImageResponse, image_to_data_url, http_error

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

logger = logging.getLogger(__name__)    

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


def handle_pipeline_error(e: Exception) -> JSONResponse:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        torch.cuda.empty_cache()
    logger.error(f"SketchToImagePipeline error: {e}")
    logger.exception(e)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=http_error("SketchToImagePipeline error"),
    )

@router.post(
    "/sketch-to-image",
    response_model=ImageResponse,
    responses=RESPONSES,
    description="Transform sketch to image.",
    operation_id="genSketchToImage",
    summary="Sketch To Image",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "sketchToImage"},
)
@router.post(
    "/sketch-to-image/",
    response_model=ImageResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def sketch_to_image(
    prompt: Annotated[
        str,
        Form(
            description=(
                "Text prompt(s) to guide image generation. Separate multiple prompts "
                "with '|' if supported by the model."
            )
        ),
    ],
    image: Annotated[
        UploadFile,
        File(description="Uploaded sketch image to generate a image from."),
    ],
    model_id: Annotated[
        str,
        Form(
            description="Hugging Face model ID used for image generation."
        ),
    ] = "",
    height: Annotated[
        int,
        Form(description="The height in pixels of the generated image."),
    ] = 512,
    width: Annotated[
        int,
        Form(description="The width in pixels of the generated image."),
    ] = 1024,
    negative_prompt: Annotated[
        str,
        Form(
            description=(
                "Text prompt(s) to guide what to exclude from image generation. "
                "Ignored if guidance_scale < 1."
            ),
        ),
    ] = "",
    num_inference_steps: Annotated[
        int,
        Form(
            description=(
                "Number of denoising steps. More steps usually lead to higher quality "
                "images but slower inference. Modulated by strength."
            ),
        ),
    ] = 8,
    controlnet_conditioning_scale: Annotated[
        float,
        Form(description="Encourages model to generate images follow the conditioning input more strictly"),
    ] = 1.0,
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

    if model_id != "" and model_id != pipeline.model_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{model_id}"
            ),
        )
    
    image = Image.open(image.file).convert("RGB")

    images = []
    has_nsfw_concept = []
    try:
        imgs, nsfw_checks = pipeline(
            prompt=prompt,
            image=image,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        images.extend(imgs)
        has_nsfw_concept.extend(nsfw_checks)
    except Exception as e:
        handle_pipeline_error(e)

    output_images = [
        {"url": image_to_data_url(img), "seed": 0, "nsfw": nsfw or False}
        for img, sd, nsfw in zip(images, [1], has_nsfw_concept)
    ]

    return {"images": output_images}