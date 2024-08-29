import logging
import os
from typing import Annotated

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.util import HTTPError, file_exceeds_max_size, http_error
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
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
    """Handles exceptions raised during audio generation.

    Args:
        e: The exception raised during audio generation.

    Returns:
        A JSONResponse with the appropriate error message and status code.
    """
    logger.error(f"Audio generation error: {str(e)}")  # Log the detailed error
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_message = "Internal server error during audio generation."

    return JSONResponse(
        status_code=status_code,
        content=http_error(error_message),
    )

@router.post(
    "/text-to-audio",
    responses=RESPONSES,
    description="Generate audio from text input.",
)
@router.post(
    "/text-to-audio/",
    responses=RESPONSES,
    include_in_schema=False,
)
async def text_to_audio(
    prompt: Annotated[
        str, Form(description="Text prompt for audio generation.")
    ],
    model_id: Annotated[
        str,
        Form(description="Hugging Face model ID used for audio generation."),
    ] = "",
    negative_prompt: Annotated[
        str, Form(description="Negative prompt for audio generation.")
    ] = None,
    seed: Annotated[
        int, Form(description="Seed for random number generation.")
    ] = None,
    num_inference_steps: Annotated[
        int, Form(description="Number of inference steps.")
    ] = 50,
    audio_length_in_s: Annotated[
        float, Form(description="Length of generated audio in seconds.")
    ] = 10.0,
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

    try:
        audio_buffers, _ = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            audio_length_in_s=audio_length_in_s,
        )

        if not audio_buffers:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=http_error("Failed to generate audio"),
            )

        # We only return the first generated audio
        audio_buffer = audio_buffers[0]

        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="generated_audio.wav"'
            },
        )

    except Exception as e:
        return handle_pipeline_error(e)