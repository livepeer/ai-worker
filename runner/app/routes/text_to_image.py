import logging
import os
import random
from typing import Annotated

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.util import (HTTPError, ImageResponse, http_error,
                             image_to_data_url)
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

router = APIRouter()

logger = logging.getLogger(__name__)


class TextToImageParams(BaseModel):
    # TODO: Make model_id and other None properties optional once Go codegen tool
    # supports OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    model_id: Annotated[
        str,
        Field(default="", description="This is the diffusion model for image generation."),
    ]
    prompt: Annotated[str, Field(description="This is the text description for the image. When prompting use + or - after the word to increase the weight of the word in generation, you can add multiple ++ or -- to increase or decrease weight.")]
    height: Annotated[int, Field(default=576, description="The height in pixels of the generated image.")]
    width: Annotated[int, Field(default=1024, description="The width in pixels of the generated image.")]
    guidance_scale: Annotated[float, Field(default=7.5, description="A higher guidance scale value encourages the model to generate images closely linked to the text prompt at the expense of lower image quality. Guidance scale is enabled when guidance_scale > 1.")]
    negative_prompt: Annotated[str, Field(default="", description="The prompt or prompts to guide what to not include in image generation. If not defined, you need to pass negative_prompt_embeds instead. Ignored when not using guidance (guidance_scale < 1).")]
    safety_check: Annotated[bool, Field(default=True, description="Classification module that estimates whether generated images could be considered offensive or harmful. Please refer to the model card for more details about a model’s potential harms.")]
    seed: Annotated[int, Field(default=None, description="The seed to set.")]
    num_inference_steps: Annotated[int, Field(default=50, description="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.")]
    num_images_per_prompt: Annotated[int, Field(default=1, description="The number of images to generate per prompt.")]


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
    params.num_images_per_prompt = 1
    for seed in seeds:
        try:
            params.seed = seed
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
