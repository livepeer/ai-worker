import logging
import os
import random
from typing import Annotated, Dict, Tuple, Union

import torch
import traceback
from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import (
    HTTPError,
    http_error,
    handle_pipeline_exception,
)
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

router = APIRouter()

logger = logging.getLogger(__name__)

# Pipeline specific error handling configuration.
PIPELINE_ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    "OutOfMemoryError": (
        "Out of memory error. Try reducing input image resolution.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
}

class LiveVideoToVideoParams(BaseModel):
    subscribe_url: Annotated[
        str,
        Field(
            ...,
            description="Source URL of the incoming stream to subscribe to.",
        ),
    ]
    publish_url: Annotated[
        str,
        Field(
            ...,
            description="Destination URL of the outgoing stream to publish.",
        ),
    ]
    model_id: Annotated[
        str,
        Field(
            default="", description="Hugging Face model ID used for image generation."
        ),
    ]
    params: Annotated[
        Dict,
        Field(
            default=None,
            description="Initial parameters for the model."
        ),
    ]

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

@router.post(
    "/live-video-to-video",
    responses=RESPONSES,
    description="Apply video-like transformations to a provided image.",
    operation_id="genLiveVideoToVideo",
    summary="Video To Video",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "liveVideoToVideo"},
)
@router.post(
    "/live-video-to-video/",
    responses=RESPONSES,
    include_in_schema=False,
)
async def live_video_to_video(
    params: LiveVideoToVideoParams,
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

    if params.model_id != "" and params.model_id != pipeline.model_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{params.model_id}."
            ),
        )

    seed = random.randint(0, 2**32 - 1)
    kwargs = {k: v for k, v in params.model_dump().items()}
    try:
        pipeline(**kwargs)
    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        logger.error(f"LiveVideoToVideoPipeline error: {e}")
        logger.error(traceback.format_exc())
        return handle_pipeline_exception(
            e,
            default_error_message="live-video-to-video pipeline error.",
            custom_error_config=PIPELINE_ERROR_CONFIG,
        )

    return {}

