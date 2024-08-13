import logging
import os
from typing import Annotated
import numpy as np

from PIL import Image
from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.util import HTTPError, MasksResponse, http_error
from fastapi import APIRouter, Depends, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

router = APIRouter()

logger = logging.getLogger(__name__)

RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}


@router.post("/sam2", response_model=MasksResponse, responses=RESPONSES)
@router.post(
    "/sam2/",
    response_model=MasksResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def SAM2(
    image: Annotated[UploadFile, File()],
    model_id: Annotated[str, Form()] = "",
    point_coords: Annotated[str, Form()] = None,
    point_labels: Annotated[str, Form()] = None,
    box: Annotated[str, Form()] = None,
    mask_input: Annotated[str, Form()] = None,
    multimask_output: Annotated[bool, Form()] = True,
    return_logits: Annotated[bool, Form()] = False,
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

    # Convert form data strings to numpy arrays
    point_coords = (
        np.fromstring(point_coords, sep=",").reshape(-1, 2) if point_coords else None
    )
    point_labels = np.fromstring(point_labels, sep=",") if point_labels else None
    box = np.fromstring(box, sep=",").reshape(-1, 4) if box else None
    mask_input = (
        np.fromstring(mask_input, sep=",").reshape(-1, 2) if mask_input else None
    )

    try:
        image = Image.open(image.file)
        masks, iou_predictions, low_res_masks = pipeline(
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
        logger.error(f"Sam2 error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=http_error("Sam2 error"),
        )

    return {
        "masks": str(masks.tolist()),
        "iou_predictions": str(iou_predictions.tolist()),
        "low_res_masks": str(low_res_masks.tolist()),
    }
