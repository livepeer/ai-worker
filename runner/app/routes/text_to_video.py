from pydantic import BaseModel
from fastapi import Depends, APIRouter, File, Form
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
    # TODO: Make model_id optional once Go codegen tool supports OAPI 3.1
    # https://github.com/deepmap/oapi-codegen/issues/373
    model_id: str = ""
    prompt: str
    height: int = 576
    width: int = 1024
    fps: int = 6
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    motion: str = ""
    seed: int = None

responses = {400: {"model": HTTPError}, 500: {"model": HTTPError}}


# TODO: Make model_id optional once Go codegen tool supports OAPI 3.1
# https://github.com/deepmap/oapi-codegen/issues/373
@router.post("/text-to-video", response_model=VideoResponse, responses=responses)
@router.post(
    "/text-to-video/",
    response_model=VideoResponse,
    responses=responses,
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
        batch_frames = pipeline(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            motion=params.motion,
            guidance_scale=params.guidance_scale,
            height=params.height,
            width=params.width,
            fps=params.fps,
            seed=params.seed,
        )
    except Exception as e:
        logger.error(f"TextToVideoPipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=500, content=http_error("TextToVideoPipeline error")
        )

    output_frames = []
    for frames in batch_frames:
        output_frames.append(
            [{"url": image_to_data_url(frame), "seed": params.seed} for frame in frames]
        )

    return {"frames": output_frames}
