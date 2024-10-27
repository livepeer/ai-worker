import av
import logging
import os
from typing import Annotated, Dict, Tuple, Union

import torch

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import (
    HTTPError,
    ObjectDetectionResponse,
    file_exceeds_max_size,
    handle_pipeline_exception,
    http_error,
    image_to_data_url,
)
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile

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


@router.post(
    "/object-detection",
    response_model=ObjectDetectionResponse,
    responses=RESPONSES,
    description="Generate annotated video(s) for object detection from the input video(s)",
    operation_id="genObjectDetection",
    summary="Object Detection",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "objectDetection"},
)
@router.post(
    "/object-detection/",
    response_model=ObjectDetectionResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def object_detection(
    video: Annotated[
        UploadFile, File(description="Uploaded video to transform with the pipeline.")
    ],
    confidence_threshold:  Annotated[
        float, Form(description="Score threshold to keep object detection predictions.")
    ] = 0.6,
    model_id: Annotated[
        str,
        Form(description="Hugging Face model ID used for transformation."),
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

    if file_exceeds_max_size(video, 50 * 1024 * 1024):
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content=http_error("File size exceeds limit"),
        )

    frames = []
    container = av.open(video)
    
    for frame in container.decode(video=0):  # Decode video frames
        frames.append(frame.to_image())  # Convert each frame to PIL image and add to list

    container.close()

    try:
        annotated_frames, confidence_scores_all_frames, labels_all_frames = pipeline(
            frames=frames,
            confidence_threshold=confidence_threshold,
        )
    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        logger.error(f"ObjectDetectionPipeline error: {e}")
        return handle_pipeline_exception(
            e,
            default_error_message="Object-detection pipeline error.",
            custom_error_config=PIPELINE_ERROR_CONFIG,
        )

    output_frames = []
    for frames in annotated_frames:
        output_frames.append(
            [
                {
                    "url": image_to_data_url(frame),
                }
                for frame in frames
            ]
        )

    return {
        "frames": output_frames,
        "confidence_scores": str(confidence_scores_all_frames),
        "labels": str(labels_all_frames),
    }
