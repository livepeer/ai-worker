from typing import Optional, Union, List
from fastapi import Depends, APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from app.pipelines.base import Pipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, extract_frames, VideoResponse, http_error
from PIL import Image
import logging
import random
import json
import os

class HTTPError(BaseModel):
    detail: str

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
    response_model=VideoResponse,
    responses=RESPONSES,
    description="Generate Lip Sync'ed video given an image and uploaded audio file or text.",
    operation_id="genLipsync",
    summary="Lipsync",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "lipsync"},
)
@router.post(
    "/lipsync/",
    response_model=VideoResponse,
    responses=RESPONSES,
    include_in_schema=False,
)

async def lipsync(
    text_input: str = Form("", description="Text input for lip-syncing."),
    tts_steering: str = Form("A male speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.", description="Prompt to steer generated voice characteristics."),
    model_id: str = Form("", description="Hugging Face model ID used for Text-to-speech."),
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
        result = pipeline(
            text_input,
            tts_steering,
            audio_file,
            image.file
        )
    except Exception as e:
        logger.error(f"LipsyncPipeline error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Internal Server Error: {str(e)}"
            },
        )
    
    if return_frames:
        frames = extract_frames(result)
        seed = random.randint(0, 1000000)
        has_nsfw_concept = [False]  # TODO: Replace with actual NSFW detection logic
        
        output_frames = [
            {
                "url": image_to_data_url(frame),
                "seed": seed,
                "nsfw": has_nsfw_concept[0],
            }
            for frame in frames
        ]
        return {"frames": [output_frames]}
    
    if os.path.exists(result):
            return FileResponse(path=result, media_type='video/mp4', filename="lipsync_video.mp4")
    else:
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"no output found for {result}"
            },
        )
