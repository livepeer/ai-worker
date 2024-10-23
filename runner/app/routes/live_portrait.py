import logging
import os
import multiprocessing
import cv2
from typing import Annotated

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import HTTPError, VideoResponse, http_error, image_to_data_url
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

logger = logging.getLogger(__name__)

RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}

def process_frame(frame):
    # Convert the frame from BGR (OpenCV format) to RGB and then to a PIL Image
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB using slicing
    pil_image = Image.fromarray(rgb_frame, 'RGB')
    return {
        "url": image_to_data_url(pil_image),  # Use the PIL Image here
        "seed": 0,  # LivePortrait doesn't use seeds
        "nsfw": False,  # LivePortrait doesn't perform NSFW checks
    }

@router.post(
    "/live-portrait",
    response_model=VideoResponse,
    responses=RESPONSES,
    description="Generate a video from a provided source image and driving video.",
)
@router.post(
    "/live-portrait/",
    response_model=VideoResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def live_portrait(
    source_image: Annotated[
        UploadFile,
        File(description="Uploaded source image to animate."),
    ],
    driving_video: Annotated[
        UploadFile,
        File(description="Uploaded driving video to guide the animation."),
    ],
    model_id: Annotated[
        str, Form(description="No model id needed as leave empty.")
    ] = "",
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
    try:
        # Save the driving video to a temporary file
        temp_video_path = "temp_driving_video.mp4"
        with open(temp_video_path, "wb") as buffer:
            buffer.write(await driving_video.read())

        temp_image_path = "temp_source_image.jpg"
        with open(temp_image_path, "wb") as buffer:
            buffer.write(await source_image.read())
        
        result_video_path = pipeline(
            source_image=temp_image_path,
            driving_info=temp_video_path
        )

        output_frames = []

        cap = cv2.VideoCapture(result_video_path)

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            futures = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                futures.append(pool.apply_async(process_frame, (frame,)))

            # Collect all processed frames directly
            output_frames = [future.get() for future in futures]
        cap.release()

    except Exception as e:
        logger.error(f"LivePortraitPipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=http_error("LivePortraitPipeline error"),
        )
    finally:
        # Clean up the temporary files
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(result_video_path):
            os.remove(result_video_path)
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    # Return frames wrapped in an outer list, adhering to the required schema
    return {"frames": [output_frames]}
