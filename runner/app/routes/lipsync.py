from typing import Optional, Union, List, Annotated
from fastapi import Depends, APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from app.pipelines.base import Pipeline
from app.dependencies import get_pipeline
from app.routes.util import HTTPError, image_to_data_url, extract_frames, http_error, VideoBinaryResponse
from PIL import Image
import logging
import random
import json
import os
import base64

router = APIRouter()

logger = logging.getLogger(__name__)
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
    "/lipsync",
    response_model=VideoBinaryResponse,
    responses=RESPONSES,
    description="Generate Lip Sync'ed video given an image and uploaded audio file or text.",
    operation_id="genLipsync",
    summary="Lipsync",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "lipsync"},
)
@router.post(
    "/lipsync/",
    response_model=VideoBinaryResponse,
    responses=RESPONSES,
    include_in_schema=False,
)

async def lipsync(
    text_input: str = Form("", description="Text input for lip-syncing."),
    tts_steering: str = Form("A male speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.", description="Prompt to steer generated voice characteristics."),
    model_id: Annotated[
        str,
        Form(description="Hugging Face model ID used for Text-to-speech."),
    ] = "",
    audio: UploadFile = File(None),
    image: UploadFile = File(...),
    return_frames: bool = Form(False, description="Set to True to return frames instead of mp4."),
    pipeline = Depends(get_pipeline)
):
    if not (text_input or audio):
        raise HTTPException(status_code=400, detail="Either text_input or audio must be provided")

    if model_id != "" and model_id != pipeline.model_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{model_id}"
            ),
        )
    
    if audio is not None:
        audio_file = audio.file
    else:
        audio_file = None

    if image is None or image.file is None:
        raise HTTPException(status_code=400, detail="Image file must be provided")

    try:
        video_file_path = pipeline(
            text_input,
            tts_steering,
            audio_file,
            image.file
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Internal Server Error: {str(e)}"
            },
        )
    
    if os.path.exists(video_file_path):
        return get_video(video_file_path)
    else:
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"no output found for {video_file_path}"
            },
        )

def get_video(video_file_path: str):
    try:
        if not os.path.exists(video_file_path):
            raise HTTPException(status_code=404, detail="Video file not found")

        with open(video_file_path, "rb") as video_file:
            binary_data = video_file.read()
            base64_video = base64.b64encode(binary_data).decode('utf-8')

        file_size = os.path.getsize(video_file_path)

        return VideoBinaryResponse(
            base64_video=base64_video,
            file_size=file_size
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while processing the video")
