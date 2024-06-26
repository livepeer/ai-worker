import logging
import os
import random
from typing import Annotated

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.util import HTTPError, VideoResponse, http_error, image_to_data_url
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

logger = logging.getLogger(__name__)

RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}


# TODO: Make model_id and other None properties optional once Go codegen tool supports
# OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
@router.post("/image-to-video", response_model=VideoResponse, responses=RESPONSES)
@router.post(
    "/image-to-video/",
    response_model=VideoResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def image_to_video(
    image: Annotated[UploadFile, File()],
    model_id: Annotated[str, Form()] = "",
    height: Annotated[int, Form()] = 576,
    width: Annotated[int, Form()] = 1024,
    fps: Annotated[int, Form()] = 6,
    motion_bucket_id: Annotated[int, Form()] = 127,
    noise_aug_strength: Annotated[float, Form()] = 0.02,
    seed: Annotated[int, Form()] = None,
    safety_check: Annotated[bool, Form()] = True,
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

    if height % 8 != 0 or width % 8 != 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"`height` and `width` have to be divisible by 8 but are {height} and "
                f"{width}."
            ),
        )

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    try:
        batch_frames, has_nsfw_concept = pipeline(
            image=Image.open(image.file).convert("RGB"),
            height=height,
            width=width,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            safety_check=safety_check,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
    except Exception as e:
        logger.error(f"ImageToVideoPipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=http_error("ImageToVideoPipeline error"),
        )

    output_frames = []
    for frames in batch_frames:
        output_frames.append(
            [
                {
                    "url": image_to_data_url(frame),
                    "seed": seed,
                    "nsfw": has_nsfw_concept[0],
                }
                for frame in frames
            ]
        )

    return {"frames": output_frames}
