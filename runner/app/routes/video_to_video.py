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
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

router = APIRouter()

logger = logging.getLogger(__name__)

# Pipeline specific error handling configuration.
PIPELINE_ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    "OutOfMemoryError": (
        "Out of memory error. Try reducing input image resolution.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
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
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}

@router.post(
    "/video-to-video",
    responses=RESPONSES,
    description="Apply video-like transformations to a provided image.",
    operation_id="genVideoToVideo",
    summary="Video To Video",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "videoToVideo"},
)
@router.post(
    "/video-to-video/",
    responses=RESPONSES,
    include_in_schema=False,
)
async def video_to_video(
    model_id: Annotated[
        str,
        Form(description="Hugging Face model ID used for image transformation."),
    ] = "",
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

    seed = random.randint(0, 2**32 - 1)
    try:
        pipeline(
            seed=seed,
        )
    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        logger.error(f"VideoToVideoPipeline error: {e}")
        logger.error(traceback.format_exc())
        return handle_pipeline_exception(
            e,
            default_error_message="Video-to-video pipeline error.",
            custom_error_config=PIPELINE_ERROR_CONFIG,
        )

    return

