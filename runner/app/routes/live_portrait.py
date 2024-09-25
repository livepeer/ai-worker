import logging
import os
import multiprocessing
import cv2
from typing import Annotated

from app.dependencies import get_pipeline
from liveportrait import Inference
from app.pipelines.base import Pipeline
from app.routes.util import HTTPError, VideoResponse, http_error, image_to_data_url
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
        str, Form(description="Hugging Face model ID used for video generation.")
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

    if model_id != "" and model_id != pipeline.model_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{model_id}"
            ),
        )

    try:
        # Save the driving video to a temporary file
        temp_video_path = "temp_driving_video.mp4"
        with open(temp_video_path, "wb") as buffer:
            buffer.write(await driving_video.read())

        temp_image_path = "temp_source_image.jpg"
        with open(temp_image_path, "wb") as buffer:
            buffer.write(await source_image.read())
        
        # Run inference to get the video output
        result_video_path, _ = Inference().run(
            source_image=temp_image_path,
            driving_video=temp_video_path)
        
        output_frames = []
        frames_batch = []  # Create a batch for frames
        cap = cv2.VideoCapture(result_video_path)

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            futures = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                futures.append(pool.apply_async(process_frame, (frame,)))

            for future in futures:
                frames_batch.append(future.get())
                if len(frames_batch) % 10 == 0:  # Process every 10th frame to reduce workload
                    output_frames.append(frames_batch)
                    frames_batch = []  # Reset the batch

        # Append any remaining frames
        if frames_batch:
            output_frames.append(frames_batch)
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
    return {"frames": output_frames}
