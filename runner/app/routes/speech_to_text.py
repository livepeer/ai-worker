from fastapi import Depends, APIRouter
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.pipelines.base import Pipeline
from app.dependencies import get_pipeline
from app.routes.util import TextResponse, HTTPError, http_error, verify_file_size
from app.pipelines.audio import AudioConverter
import logging
import random
import os
from typing import Annotated
from fastapi import Depends, APIRouter, UploadFile, File, Form

router = APIRouter()

logger = logging.getLogger(__name__)

responses = {400: {"model": HTTPError}, 500: {"model": HTTPError}}

@router.post("/speech-to-text", response_model=TextResponse, responses=responses)
@router.post("/speech-to-text/", response_model=TextResponse, include_in_schema=False)
async def speech_to_text(
    audio: Annotated[UploadFile, File()],
    model_id: Annotated[str, Form()] = "",
    seed: Annotated[int, Form()] = None,
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    verify_file_size(audio, 50 * 1024 * 1024)
    auth_token = os.environ.get("AUTH_TOKEN")
    if auth_token:
        if not token or token.credentials != auth_token:
            return JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
                content=http_error("Invalid bearer token"),
            )

    if model_id != "" and model_id != pipeline.model_id:
        return JSONResponse(
            status_code=400,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{model_id}"
            ),
        )

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    try:
        # Convert MP4 files to improve compatibility
        if audio.filename.endswith(".m4a") or audio.filename.endswith(".mp4"):
            conv = AudioConverter()
            converted_bytes = conv.mp4_to_mp3(audio)
            audio.file.seek(0)
            audio.file.write(converted_bytes)
            audio.file.seek(0)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=http_error(f"Error processing audio file: {str(e)}"),
        )

    try:
        return pipeline(
            audio=audio.file.read()
        )
    except Exception as e:
        status_code = 500
        if "Soundfile is either not in the correct format or is malformed" in str(e):
            status_code = 400

        return JSONResponse(
            status_code=status_code,
            content=http_error(f"Error processing audio file: {str(e)}"),
        )
