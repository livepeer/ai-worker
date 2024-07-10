from typing import Optional
from fastapi import Depends, APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from app.pipelines.base import Pipeline
from app.dependencies import get_pipeline
import logging
import random
import json
import os

class HTTPError(BaseModel):
    detail: str

router = APIRouter()

logger = logging.getLogger(__name__)

responses = {
    400: {"content": {"application/json": {"schema": HTTPError.schema()}}},
    500: {"content": {"application/json": {"schema": HTTPError.schema()}}},
    200: {"content": {"video/mp4": {}}}
}

@router.post("/lipsync", responses=responses)
async def lipsync(
    text_input: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    image: UploadFile = File(...),
    pipeline: Pipeline = Depends(get_pipeline),
):
    if not text_input and not audio:
        raise HTTPException(status_code=400, detail="Either text_input or audio must be provided")

    try:
        output_video_path = pipeline(
            text_input,
            audio.file,
            image.file,
        )
    except Exception as e:
        logger.error(f"LipsyncPipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Internal Server Error: {str(e)}"
            },
        )

    if os.path.exists(output_video_path):
        return FileResponse(path=output_video_path, media_type='video/mp4', filename="lipsync_video.mp4")
    else:
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"no output found for {output_video_path}"
            },
        )

