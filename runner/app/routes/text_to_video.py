from fastapi import APIRouter, Depends, Form, status, Depends, APIRouter, Form
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.pipelines.base import Pipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, VideoResponse, HTTPError, http_error
from typing import Annotated
import logging
import random
import os

router = APIRouter()

logger = logging.getLogger(__name__)

class TextToVideoParams(BaseModel):
    # TODO: Make model_id and other None properties optional once Go codegen tool
    # supports OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    model_id: Annotated[
        str,
        Field(
            default="", description="Hugging Face model ID used for video generation."
        ),
    ]
    prompt: Annotated[
        str,
        Field(
            description=(
                "Text prompt(s) to guide video generation. Separate multiple prompts "
                "with '|' if supported by the model."
            )
        ),
    ]
    height: Annotated[
        int,
        Field(default=576, description="The height in pixels of the generated video."),
    ]
    width: Annotated[
        int,
        Field(default=1024, description="The width in pixels of the generated video."),
    ]
    fps: Annotated[
        int,
        Field(default=8, description="The frames per second of the generated video."),
    ]
    motion_bucket_id: Annotated[
        int,
        Field(default=127, description=(
            "Used for conditioning the amount of motion for the generation. The "
            "higher the number the more motion will be in the video."
        )),
    ]
    noise_aug_strength: Annotated[
        float,
        Field(default=0.02, description=(
            "Amount of noise added to the conditioning image. Higher values reduce "
            "resemblance to the conditioning image and increase motion."
        )),
    ]
    guidance_scale: Annotated[
        float,
        Field(
            default=7.5,
            description=(
                "Encourages model to generate videos closely linked to the text prompt "
                "(higher values may reduce image quality)."
            ),
        ),
    ]
    negative_prompt: Annotated[
        str,
        Field(
            default="",
            description=(
                "Text prompt(s) to guide what to exclude from video generation. "
                "Ignored if guidance_scale < 1."
            ),
        ),
    ]
    safety_check: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Perform a safety check to estimate if generated videos could be "
                "offensive or harmful."
            ),
        ),
    ]
    seed: Annotated[
        int, Field(default=None, description="Seed for random number generation."),
    ]
    num_inference_steps: Annotated[
        int,
        Field(
            default=50,
            description=(
                "Number of denoising steps. More steps usually lead to higher quality "
                "images but slower inference. Modulated by strength."
            ),
        ),
    ]

RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}

@router.post(
    "/text-to-video",
    response_model=VideoResponse,
    responses=RESPONSES,
    description="Generate videos from text prompts.",
)
@router.post(
    "/text-to-video/",
    response_model=VideoResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def text_to_video(
    params: TextToVideoParams,
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

    if params.model_id != "" and params.model_id != pipeline.model_id:
        return JSONResponse(
            status_code=400,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with {params.model_id}"
            ),
        )

    if params.height % 8 != 0 or params.width % 8 != 0:
        return JSONResponse(
            status_code=400,
            content=http_error(
                f"`height` and `width` have to be divisible by 8 but are {params.height} and {params.width}."
            ),
        )

    if params.seed is None:
        params.seed = random.randint(0, 2**32 - 1)

    try:
        kwargs = {k: v for k, v in params.model_dump().items() if k != "model_id"}
        batch_frames = pipeline(**kwargs)
    except Exception as e:
        logger.error(f"TextToVideoPipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=http_error("TextToVideoPipeline error"),
        )

    output_frames = []
    for frames in batch_frames:
        output_frames.append(
            [
                {
                    "url": image_to_data_url(frame),
                    "seed": params.seed,
                    "nsfw": False,
                }
                for frame in frames
            ]
        )

    return {"frames": output_frames}
