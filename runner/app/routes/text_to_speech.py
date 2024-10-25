import logging
import os
import base64
import torch

from typing import Annotated, Dict, Tuple, Union
from fastapi import Depends, APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from app.pipelines.base import Pipeline
from app.dependencies import get_pipeline
from app.routes.utils import (
    HTTPError,
    http_error,
    handle_pipeline_exception,
    EncodedFileResponse,
)

router = APIRouter()

logger = logging.getLogger(__name__)

# Pipeline specific error handling configuration.
PIPELINE_ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    # Specific error types.
    "OutOfMemoryError": (
        "Out of memory error. Try reducing output image resolution.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
}


class TextToSpeechParams(BaseModel):
    # TODO: Make model_id and other None properties optional once Go codegen tool
    # supports OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    model_id: Annotated[
        str,
        Field(
            default="",
            description="Hugging Face model ID used for text to speech generation.",
        ),
    ]
    text_input: Annotated[
        str,
        Field(
            default="",
            description=("Text input for speech generation"),
        ),
    ]
    description: Annotated[
        str,
        Field(
            default=(
                "A male speaker delivers a slightly expressive and animated speech "
                "with a moderate speed and pitch."
            ),
            description=("Description of speaker to steer text to speech generation"),
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


# TODO: Make model_id and other None properties optional once Go codegen tool supports
# OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
@router.post(
    "/text-to-speech",
    response_model=EncodedFileResponse,
    responses=RESPONSES,
    description=(
        "Generate text-to-speech audio file as determined by text_input and "
        "tts_steering description of voice."
    ),
    operation_id="genTextToSpeech",
    summary="text-to-speech",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "textToSpeech"},
)
@router.post(
    "/text-to-speech/",
    response_model=EncodedFileResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def text_to_speech(
    params: TextToSpeechParams, pipeline: Pipeline = Depends(get_pipeline)
):
    if not (params.text_input):
        raise HTTPException(status_code=400, detail="text_input must be provided")

    if params.model_id != "" and params.model_id != pipeline.model_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{params.model_id}"
            ),
        )

    try:
        audio_file_path = pipeline(params)
    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        logger.error(f"TextToSpeechPipeline error: {e}")
        return handle_pipeline_exception(
            e,
            default_error_message="Text-to-image pipeline error.",
            custom_error_config=PIPELINE_ERROR_CONFIG,
        )

    if os.path.exists(audio_file_path):
        return encode_file(audio_file_path)
    else:
        return JSONResponse(
            status_code=400,
            content={"detail": f"no output found for {audio_file_path}"},
        )


def encode_file(file_path: str):
    try:
        # Read the binary audio file and encode it as base64
        with open(file_path, "rb") as file:
            binary_data = file.read()
            base64_audio = base64.b64encode(binary_data).decode("utf-8")

        # Get the file size
        file_size = os.path.getsize(file_path)

        # Return the response model with the base64-encoded video
        return EncodedFileResponse(base64_data=base64_audio, file_size=file_size)

    except Exception as e:
        # Log or print the error for debugging purposes
        print(f"An error occurred while processing the video: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred while processing the video"
        )
