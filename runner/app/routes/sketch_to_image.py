import logging
import os
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.pipelines.utils.audio import AudioConversionError
from app.routes.util import HTTPError, TextResponse, file_exceeds_max_size, http_error

router = APIRouter()

logger = logging.getLogger(__name__)

RESPONSES = {
    status.HTTP_200_OK: {
        "content": {
            "application/json": {}
        },
    },
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}


def handle_pipeline_error(e: Exception) -> JSONResponse:
    """Handles exceptions raised during audio processing.

    Args:
        e: The exception raised during audio processing.

    Returns:
        A JSONResponse with the appropriate error message and status code.
    """
    logger.error(f"Audio processing error: {str(e)}")  # Log the detailed error
    if "Soundfile is either not in the correct format or is malformed" in str(
        e
    ) or isinstance(e, AudioConversionError):
        status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        error_message = "Unsupported audio format or malformed file."
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_message = "Internal server error during audio processing."

    return JSONResponse(
        status_code=status_code,
        content=http_error(error_message),
    )


@router.post(
    "/sketch-to-image",
    response_model=TextResponse,
    responses=RESPONSES,
    description="Transform sketch to image.",
    operation_id="genSketchToImage",
    summary="Sketch To Image",
    tags=["generate"],
)
@router.post(
    "/sketch-to-image/",
    response_model=TextResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def sketch_to_image():
    pass