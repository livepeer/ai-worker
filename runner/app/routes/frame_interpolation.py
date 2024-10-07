# app/routes/film_interpolate.py

import logging
import os
import torch
import glob
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile

from app.dependencies import get_pipeline
from app.pipelines.frame_interpolation import FILMPipeline
from app.pipelines.utils.utils import DirectoryReader, DirectoryWriter, get_torch_device, get_model_dir
from app.routes.util import HTTPError, ImageResponse, http_error, image_to_data_url

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

logger = logging.getLogger(__name__)

RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}

@router.post("/frame_interpolation", response_model=ImageResponse, responses=RESPONSES)
@router.post(
    "/frame_interpolation/",
    response_model=ImageResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def frame_interpolation(
    model_id: Annotated[str, Form()],
    image1: Annotated[UploadFile, File()]=None,
    image2: Annotated[UploadFile, File()]=None,
    image_dir: Annotated[str, Form()]="",
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
    film_pipeline.to(device=get_torch_device(),dtype=torch.float16)

    # Prepare directories for input and output
    temp_input_dir = "temp_input"
    temp_output_dir = "temp_output"
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)

    try:
        if os.path.isdir(image_dir):
            if image1 and image2:
                logger.info("Both directory and individual images provided. Directory will be used, and images will be ignored.")
            reader = DirectoryReader(image_dir)
        else:
            if not (image1 and image2):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=http_error("Either a directory or two images must be provided."),
                )

            image1_path = os.path.join(temp_input_dir, "0.png")
            image2_path = os.path.join(temp_input_dir, "1.png")

            with open(image1_path, "wb") as f:
                f.write(await image1.read())
            with open(image2_path, "wb") as f:
                f.write(await image2.read())

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

        output_images = [{"url": image_to_data_url(frame),"seed":0, "nsfw":False} for frame in output_frames]

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

    return {"images": output_images}
