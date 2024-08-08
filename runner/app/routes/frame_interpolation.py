# app/routes/film_interpolate.py

import logging
import os
import torch
import glob
import cv2
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile

from app.dependencies import get_pipeline
from app.pipelines.frame_interpolation import FILMPipeline
from app.pipelines.utils.utils import DirectoryReader, DirectoryWriter, get_torch_device, video_shredder
from app.routes.util import HTTPError, VideoResponse, http_error, image_to_data_url

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

logger = logging.getLogger(__name__)

RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}

@router.post("/frame-interpolation", response_model=VideoResponse, responses=RESPONSES)
@router.post(
    "/frame-interpolation/",
    response_model=VideoResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def frame_interpolation(
    model_id: Annotated[str, Form()] = "",
    video: Annotated[UploadFile, File()]=None,
    inter_frames: Annotated[int, Form()] = 2,
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

    # Initialize FILMPipeline
    film_pipeline = FILMPipeline(model_id)
    film_pipeline.to(device=get_torch_device(), dtype=torch.float16)

    # Prepare directories for input and output
    temp_input_dir = "temp_input"
    temp_output_dir = "temp_output"
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)

    try:
        # Extract frames from video
        video_data = await video.read()
        frames = video_shredder(video_data, is_file_path=False)

        # Save frames to temporary directory
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_input_dir, f"{i}.png")
            cv2.imwrite(frame_path, frame)

        # Create DirectoryReader and DirectoryWriter
        reader = DirectoryReader(temp_input_dir)
        writer = DirectoryWriter(temp_output_dir)

        # Perform interpolation
        film_pipeline(reader, writer, inter_frames=inter_frames)
        writer.close()
        reader.reset()

        # Collect output frames
        output_frames = []
        for frame_path in sorted(glob.glob(os.path.join(temp_output_dir, "*.png"))):
            frame = Image.open(frame_path)
            output_frames.append(frame)
    # Wrap output frames in a list of batches (with a single batch in this case)
        output_images = [[{"url": image_to_data_url(frame), "seed": 0, "nsfw": False} for frame in output_frames]]

    except Exception as e:
        logger.error(f"FILMPipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=http_error("FILMPipeline error"),
        )
    finally:
        # Clean up temporary directories
        for file_path in glob.glob(os.path.join(temp_input_dir, "*")):
            os.remove(file_path)
        os.rmdir(temp_input_dir)
        for file_path in glob.glob(os.path.join(temp_output_dir, "*")):
            os.remove(file_path)
        os.rmdir(temp_output_dir)

    return {"frames": output_images}
