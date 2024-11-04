import base64
import json
import logging
import os
import zstd
from typing import Annotated

import numpy as np
import torch
from pydantic import ValidationError
from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import (
    HTTPError,
    VideoSegmentResponse,
    VideoSegmentationItem,
    http_error,
    json_str_to_np_array,
    handle_pipeline_exception
)
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile
from typing import List

from app.utils.errors import InferenceError

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

logger = logging.getLogger(__name__)

RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}


# TODO: Make model_id and other None properties optional once Go codegen tool supports
# OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373.
@router.post(
    "/segment-anything-2-video",
    response_model=VideoSegmentResponse,
    responses=RESPONSES,
    operation_id="genSegmentAnything2Video",
    description="Segment objects in an video.",
    summary="Segment Anything 2 Video",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "segmentAnything2Video"},
)
@router.post(
    "/segment-anything-2-video/",
    response_model=VideoSegmentResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def segment_anything_2_video(
    media_file: Annotated[
        UploadFile, File(description="Media file to segment.", media_type="video/mp4")
    ],
    model_id: Annotated[
        str, Form(description="Hugging Face model ID used for image generation.")
    ] = "",
    point_coords: Annotated[
        str,
        Form(
            description=(
                "Nx2 array of point prompts to the model, where each point is in (X,Y) "
                "in pixels."
            )
        ),
    ] = None,
    point_labels: Annotated[
        str,
        Form(
            description=(
                "Labels for the point prompts, where 1 indicates a foreground point "
                "and 0 indicates a background point."
            )
        ),
    ] = None,
    box: Annotated[
        str,
        Form(
            description=(
                "A length 4 array given as a box prompt to the model, in XYXY format."
            )
        ),
    ] = None,
    mask_input: Annotated[
        str,
        Form(
            description=(
                "A low-resolution mask input to the model, typically from a previous "
                "prediction iteration, with the form 1xHxW (H=W=256 for SAM)."
            )
        ),
    ] = None,
    multimask_output: Annotated[
        bool,
        Form(
            description=(
                "If true, the model will return three masks for ambiguous input "
                "prompts, often producing better masks than a single prediction."
            )
        ),
    ] = True,
    return_logits: Annotated[
        bool,
        Form(
            description=(
                "If true, returns un-thresholded mask logits instead of a binary mask."
            )
        ),
    ] = True,
    normalize_coords: Annotated[
        bool,
        Form(
            description=(
                "If true, the point coordinates will be normalized to the range [0,1], "
                "with point_coords expected to be with respect to image dimensions."
            )
        ),
    ] = True,
    frame_idx : Annotated[
        int, 
        Form(description="Frame index reference for (required video file input only)")
    ] = 0,
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
                f"pipeline configured with {pipeline.model_id} but called with {model_id}"
            ),
        )

    try:
        point_coords = json_str_to_np_array(point_coords, var_name="point_coords")
        point_labels = json_str_to_np_array(point_labels, var_name="point_labels")
        box = json_str_to_np_array(box, var_name="box")
        mask_input = json_str_to_np_array(mask_input, var_name="mask_input")
    except ValueError as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(str(e)),
        )

    try:
        video_segment_responses = []

        for _, _, out_mask_logits in pipeline(
            media_file,
            media_type="video",
            frame_idx=frame_idx,
            points=point_coords,
            labels=point_labels
        ):
            bool_mask = (out_mask_logits > 0.5).cpu().numpy()
            bool_mask_bytes = bool_mask.astype(np.uint8).tobytes()
            
            compressed_data = zstd.compress(bool_mask_bytes, 1)
            encoded_data = base64.b64encode(compressed_data).decode('utf-8')
            
            video_segment_responses.append(
                VideoSegmentationItem(
                    mask=encoded_data,
                    shape=bool_mask.shape
            ))
                
        return VideoSegmentResponse(
            VideoFrames=video_segment_responses
        )

    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        logger.error(f"SegmentAnything2Video pipeline error: {e}")
        return handle_pipeline_exception(
            e,
            default_error_message="Segment-anything-2 pipeline error.",
            custom_error_config=PIPELINE_ERROR_CONFIG,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=http_error(f"Segment Anything 2 error: {e}"),
        )
