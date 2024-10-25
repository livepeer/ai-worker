import logging
import os
from typing import Annotated, Dict, Tuple, Union

import torch
from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import (
    HTTPError,
    http_error,
    handle_pipeline_exception,
)
from fastapi import APIRouter, Depends, Form, Response, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

# Pipeline specific error handling configuration
PIPELINE_ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    # Specific error types
    "OutOfMemoryError": (
        "Out of memory error. Try reducing audio duration.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    ),
    "ValueError": (
        None,  # Use the error message from the exception
        status.HTTP_400_BAD_REQUEST,
    ),
}

RESPONSES = {
    status.HTTP_200_OK: {
        "content": {
            "audio/wav": {
                "schema": {
                    "type": "string",
                    "format": "binary"
                }
            },
        },
        "description": "Successfully generated audio"
    },
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}

@router.post(
    "/text-to-audio",
    responses=RESPONSES,
    description="Generate audio from text prompts using Stable Audio.",
    operation_id="genTextToAudio",
    summary="Text To Audio",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "textToAudio"},
)
@router.post(
    "/text-to-audio/",
    responses=RESPONSES,
    include_in_schema=False,
)
async def text_to_audio(
    prompt: Annotated[
        str,
        Form(description="Text prompt for audio generation."),
    ],
    model_id: Annotated[
        str,
        Form(description="Hugging Face model ID used for audio generation."),
    ] = "",
    duration: Annotated[
        float,
        Form(description="Duration of generated audio in seconds (between 1 and 30 seconds)."),
    ] = 5.0,
    num_inference_steps: Annotated[
        int,
        Form(description="Number of denoising steps. More steps usually lead to higher quality audio but slower inference."),
    ] = 10,
    guidance_scale: Annotated[
        float,
        Form(description="Scale for classifier-free guidance. Higher values result in audio that better matches the prompt but may be lower quality."),
    ] = 3.0,
    negative_prompt: Annotated[
        str,
        Form(description="Text prompt to guide what to exclude from audio generation."),
    ] = None,
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    # Validate auth token if configured
    auth_token = os.environ.get("AUTH_TOKEN")
    if auth_token:
        if not token or token.credentials != auth_token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
                content=http_error("Invalid bearer token."),
            )

    # Validate model ID matches pipeline
    if model_id != "" and model_id != pipeline.model_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{model_id}."
            ),
        )

    # Validate duration
    if duration < 1.0 or duration > 30.0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                "Duration must be between 1 and 30 seconds."
            ),
        )

    try:
        # Generate audio
        audio_data, audio_format = pipeline(
            prompt=prompt,
            duration=duration,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
        )

        # Return audio file
        return Response(
            content=audio_data,
            media_type=f"audio/{audio_format}",
            headers={
                "Content-Disposition": f"attachment; filename=generated_audio.{audio_format}"
            }
        )

    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        logger.error(f"TextToAudio pipeline error: {e}")
        return handle_pipeline_exception(
            e,
            default_error_message="Text-to-audio pipeline error.",
            custom_error_config=PIPELINE_ERROR_CONFIG,
        )