import logging
import os
import random
from typing import Annotated

import torch
from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import HTTPError, ImageResponse, http_error, image_to_data_url
from app.utils.errors import InferenceError, OutOfMemoryError
from app.pipelines.utils.utils import LoraLoadingError
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

logger = logging.getLogger(__name__)


def handle_pipeline_error(e: Exception) -> JSONResponse:
    """Handles exceptions raised during image-to-image pipeline processing.

    Args:
        e: The exception raised during image-to-image processing.

    Returns:
        A JSONResponse with the appropriate error message and status code.
    """
    logger.error(
        f"ImageToImagePipeline pipeline error: {str(e)}"
    )  # Log the detailed error
    logger.exception(e) # TODO: Check if needed.
    if "CUDA out of memory" in str(e) or isinstance(e, OutOfMemoryError) or isinstance(torch.cuda.OutOfMemoryError): # TODO: simplify condition.
        status_code = status.HTTP_400_BAD_REQUEST
        error_message = "Out of memory error. Try reducing input image resolution."
        torch.cuda.empty_cache()
    elif isinstance(e, LoraLoadingError):
        status_code = status.HTTP_400_BAD_REQUEST
        error_message = str(e)
    elif isinstance(e, InferenceError):
        status_code = status.HTTP_400_BAD_REQUEST
        error_message = str(e)
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_message = "Image-to-image pipeline error."

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
    "/image-to-image",
    response_model=ImageResponse,
    responses=RESPONSES,
    description="Apply image transformations to a provided image.",
    operation_id="genImageToImage",
    summary="Image To Image",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "imageToImage"},
)
@router.post(
    "/image-to-image/",
    response_model=ImageResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def image_to_image(
    prompt: Annotated[
        str,
        Form(description="Text prompt(s) to guide image generation."),
    ],
    image: Annotated[
        UploadFile,
        File(description="Uploaded image to modify with the pipeline."),
    ],
    model_id: Annotated[
        str,
        Form(description="Hugging Face model ID used for image generation."),
    ] = "",
    loras: Annotated[
        str,
        Form(
            description=(
                "A LoRA (Low-Rank Adaptation) model and its corresponding weight for "
                'image generation. Example: { "latent-consistency/lcm-lora-sdxl": '
                '1.0, "nerijs/pixel-art-xl": 1.2}.'
            )
        ),
    ] = "",
    strength: Annotated[
        float,
        Form(
            description=(
                "Degree of transformation applied to the reference image (0 to 1)."
            )
        ),
    ] = 0.8,
    guidance_scale: Annotated[
        float,
        Form(
            description=(
                "Encourages model to generate images closely linked to the text prompt "
                "(higher values may reduce image quality)."
            )
        ),
    ] = 7.5,
    image_guidance_scale: Annotated[
        float,
        Form(
            description=(
                "Degree to which the generated image is pushed towards the initial "
                "image."
            )
        ),
    ] = 1.5,
    negative_prompt: Annotated[
        str,
        Form(
            description=(
                "Text prompt(s) to guide what to exclude from image generation. "
                "Ignored if guidance_scale < 1."
            )
        ),
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
    ] = 100,  # NOTE: Hardcoded due to varying pipeline values.
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
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{model_id}."
            ),
        )

    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    seeds = [seed + i for i in range(num_images_per_prompt)]

    image = Image.open(image.file).convert("RGB")

    # TODO: Process one image at a time to avoid CUDA OEM errors. Can be removed again
    # once LIV-243 and LIV-379 are resolved.
    images = []
    has_nsfw_concept = []
    for seed in seeds:
        try:
            imgs, nsfw_checks = pipeline(
                prompt=prompt,
                image=image,
                strength=strength,
                loras=loras,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                negative_prompt=negative_prompt,
                safety_check=safety_check,
                seed=seed,
                num_images_per_prompt=1,
                num_inference_steps=num_inference_steps,
            )
        except Exception as e:
            return handle_pipeline_error(e)
        images.extend(imgs)
        has_nsfw_concept.extend(nsfw_checks)

    # TODO: Return None once Go codegen tool supports optional properties
    # OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    output_images = [
        {"url": image_to_data_url(img), "seed": sd, "nsfw": nsfw or False}
        for img, sd, nsfw in zip(images, seeds, has_nsfw_concept)
    ]

    return {"images": output_images}
