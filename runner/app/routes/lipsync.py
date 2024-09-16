from typing import Optional, Union, List
from fastapi import Depends, APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from app.pipelines.base import Pipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, extract_frames, VideoResponse
from PIL import Image
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
    # 200: {
    #     "content": {
    #         "video/mp4": {},
    #         "application/json": {"schema": VideoResponse.schema()},
    #     }
    # }
}

@router.post("/lipsync", responses=responses)
async def lipsync(
    text_input: Optional[str] = Form(None),
    audio: UploadFile = File(None),
    image: UploadFile = File(...),
    return_frames: Optional[bool] = Form(False, description="Set to True to return frames instead of mp4"),
    pipeline: Pipeline = Depends(get_pipeline),
):
    if not text_input and not audio:
        raise HTTPException(status_code=400, detail="Either text_input or audio must be provided")
    
    if audio is not None:
        audio_file = audio.file
    else:
        audio_file = None

    if image is None or image.file is None:
        raise HTTPException(status_code=400, detail="Image file must be provided")


    try:
        result = pipeline(
            text_input,
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
