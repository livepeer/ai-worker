import json
import numpy as np
import logging
import os
from typing import Annotated

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.util import (
    HTTPError,
    InferenceError,
    MasksResponse,
    http_error,
    json_str_to_np_array,
)
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image

router = APIRouter()

logger = logging.getLogger(__name__)

RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}


@router.post("/segment-anything-2", response_model=MasksResponse, responses=RESPONSES)
@router.post(
    "/segment-anything-2/",
    response_model=MasksResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def SegmentAnything2(
    image: Annotated[UploadFile, File()],
    model_id: Annotated[str, Form()] = "",
    point_coords: Annotated[str, Form()] = None,
    point_labels: Annotated[str, Form()] = None,
    box: Annotated[str, Form()] = None,
    mask_input: Annotated[str, Form()] = None,
    multimask_output: Annotated[bool, Form()] = True,
    return_logits: Annotated[bool, Form()] = True,
    normalize_coords: Annotated[bool, Form()] = True,
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

    try:
        image = Image.open(image.file)
        masks, scores, low_res_mask_logits = pipeline(
            image,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
            normalize_coords=normalize_coords,
        )
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
            content=http_error("Segment Anything 2 error"),
        )

    # Return masks sorted by descending score as JSON.
    sorted_ind = np.argsort(scores)[::-1]
    return {
        "masks":  str(masks[sorted_ind].tolist()),
        "scores":  str(scores[sorted_ind].tolist()),
        "logits":  str(low_res_mask_logits[sorted_ind].tolist()),
    }
