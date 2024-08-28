import logging
import os
from typing import Annotated

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.pipelines.utils.audio import AudioConversionError
from app.routes.util import HTTPError, TextResponse, file_exceeds_max_size, http_error
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

router = APIRouter()

logger = logging.getLogger(__name__)

RESPONSES = {
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
    "/audio-to-text",
    response_model=TextResponse,
    responses=RESPONSES,
    description="Transcribe audio files to text.",
)
@router.post(
    "/audio-to-text/",
    response_model=TextResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def audio_to_text(
    audio: Annotated[
        UploadFile, File(description="Uploaded audio file to be transcribed.")
    ],
    model_id: Annotated[
        str,
        Form(description="Hugging Face model ID used for transcription."),
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

    if file_exceeds_max_size(audio, 50 * 1024 * 1024):
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content=http_error("File size exceeds limit"),
        )

    try:
        return pipeline(audio=audio)
    except Exception as e:
        return handle_pipeline_error(e)
