import logging
import os
from typing import Annotated

import numpy as np
from pydantic import ValidationError
from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.util import (
    HTTPError,
    InferenceError,
    MasksResponse,
    VideoSegmentResponse,
    http_error,
    json_str_to_np_array,
)
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile
from typing import List

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
    response_model=MasksResponse,
    responses=RESPONSES,
    description="Segment objects in an image.",
)
@router.post(
    "/segment-anything-2-video/",
    response_model=MasksResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def segment_anything_2_video(
    media_file: Annotated[
        # UploadFile, File(description="Image to segment.", media_type="image/*")
        UploadFile, File(description="Media file to segment.", media_type="image/*,video/mp4")
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
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{model_id}"
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

    supported_video_types = ["video/mp4"]
    supported_image_types = ["image/jpeg", "image/png", "image/jpg"]
    supported_media_types = supported_image_types + supported_video_types

    try:
        video_segments = {}

        for out_frame_idx, out_obj_ids, out_mask_logits in pipeline(
            media_file,
            media_type="video",
            frame_idx=frame_idx,
            points=point_coords,
            labels=point_labels
        ):
            # Collect the data for each frame
            segment_data = {
                "frame_idx":out_frame_idx, #type int
                "obj_ids":out_obj_ids, #type list
                "mask_logits": [mask.cpu().numpy().tolist() for mask in out_mask_logits]
            }
            if out_frame_idx not in video_segments:
                video_segments[out_frame_idx] = {"obj_ids": [], "mask_logits": []}
            
            video_segments[out_frame_idx]["obj_ids"].extend(out_obj_ids)
            video_segments[out_frame_idx]["mask_logits"].extend(
                [mask.cpu().numpy().tolist() for mask in out_mask_logits]
            )

        video_segment_responses = []

        for frame_idx, segments in video_segments.items():
            obj_ids = segments["obj_ids"]
            mask_logits = segments["mask_logits"]
            try:
                # Create a dictionary to map obj_id to its corresponding mask_logits
                mask_logits_dict = {obj_id: mask_logits[i] for i, obj_id in enumerate(obj_ids)}
                mask_logits_list = [mask_logits_dict[obj_id] for obj_id in obj_ids]

                segment_data = VideoSegmentResponse(
                    frame_idx=frame_idx,
                    obj_ids=obj_ids,
                    mask_logits=mask_logits_list
                )
                video_segment_responses.append(segment_data)
            except ValidationError as e:
                print(f"Validation error for frame {frame_idx}: {e}")

        # Print video_segment_responses to inspect its structure
        # print("video_segment_responses:", video_segment_responses)

        # TODO: Return response in some usable format
        return MasksResponse(
            masks="",
            scores="",
            logits="",
            video_segments=video_segment_responses
        )

        # else:
        #     raise InferenceError(f"Unsupported media type: {media_file.content_type}")

    except Exception as e:
        logger.error(f"Segment Anything 2 error: {e}")
        logger.exception(e)
        if isinstance(e, InferenceError):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=http_error(str(e)),
            )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=http_error(f"Segment Anything 2 error: {e}"),
        )

    # # Return masks sorted by descending score as string.
    # sorted_ind = np.argsort(scores)[::-1]
    # return {
    #     "masks": str(masks[sorted_ind].tolist()),
    #     "scores": str(scores[sorted_ind].tolist()),
    #     "logits": str(low_res_mask_logits[sorted_ind].tolist()),
    # }
