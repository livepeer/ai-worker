# app/routes/film_interpolate.py
import os
import cv2
import glob
import logging

from typing import Annotated, Dict, Tuple, Union
from PIL import Image, ImageFile
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.pipelines.base import Pipeline
from app.dependencies import get_pipeline
from app.pipelines.utils.utils import DirectoryReader, DirectoryWriter, video_shredder
from app.routes.utils import HTTPError, VideoResponse, http_error, image_to_data_url, handle_pipeline_exception

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

logger = logging.getLogger(__name__)

# Pipeline specific error handling configuration.
PIPELINE_ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    # Specific error types.
    "OutOfMemoryError": (
        "Out of memory error. Try reducing input image resolution.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
}

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
    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"model": HTTPError},
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
    model_id: Annotated[str, Form(description="Currently there is only one model used `film_net_fp16.pt`.")] = "",
    video: Annotated[UploadFile, File(description="Video file of any arbitrary length.")]= None,
    inter_frames: Annotated[int, Form(description="Number of frames to create as the intermediate frames.")] = 2,
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
            try:
                frame_path = os.path.join(temp_input_dir, f"{i}.png")
                cv2.imwrite(frame_path, frame)
            except Exception as e:
                logger.error(f"Error saving frame {i}: {e}")

        # Create DirectoryReader and DirectoryWriter
        reader = DirectoryReader(temp_input_dir)
        writer = DirectoryWriter(temp_output_dir)

        # Initialize FILMPipeline
        pipeline(reader=reader, writer=writer, inter_frames=inter_frames)

        writer.close()
        reader.reset()

        # Collect output frames
        output_frames = []
        path_to_file = sorted(
            glob.glob(os.path.join(temp_output_dir, "*")),
            key=lambda x: (int(os.path.basename(x).split(".")[0]), x)
        )
        for frame_in_path in path_to_file:
            try:
                frame = Image.open(frame_in_path)
                output_frames.append(frame)
            except Exception as e:
                logger.error(f"Error reading frame {frame_in_path}: {e}")

        # Wrap output frames in a list of batches (with a single batch in this case)
        output_images = [[{"url": image_to_data_url(frame), "seed": 0, "nsfw": False} for frame in output_frames]]

    except Exception as e:
        logger.error(f"FILMPipeline error: {e}")
        return handle_pipeline_exception(
            e,
            default_error_message="frame-interpolation pipeline error.",
            custom_error_config=PIPELINE_ERROR_CONFIG,
        )
    finally:
        # Clean up temporary directories
        for path_to_file in glob.glob(os.path.join(temp_input_dir, "*")):
            os.remove(path_to_file)
        os.rmdir(temp_input_dir)
        for path_to_file in glob.glob(os.path.join(temp_output_dir, "*")):
            os.remove(path_to_file)
        os.rmdir(temp_output_dir)

    return {"frames": output_images}
