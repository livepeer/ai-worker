import logging
import os
from typing import Annotated

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.util import HTTPError, TextResponse, file_exceeds_max_size, http_error
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from PIL import Image

router = APIRouter()

logger = logging.getLogger(__name__)

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
    "/image-to-text",
    response_model=TextResponse,
    responses=RESPONSES,
    description="Transform image files to text.",
    operation_id="genImageToText",
    summary="Image To Text",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "imageToText"},
)
@router.post(
    "/image-to-text/",
    response_model=TextResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def image_to_text(
    prompt: Annotated[
        str, Form(description="Text prompt(s) to guide transformation."),
    ],
    image: Annotated[
        UploadFile, File(description="Uploaded image to transform with the pipeline.")
    ],
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

    if file_exceeds_max_size(image, 50 * 1024 * 1024):
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content=http_error("File size exceeds limit"),
        )

    image = Image.open(image.file).convert("RGB")
    try:
        return JSONResponse(content={"text": pipeline(prompt=prompt, image=image)})
    except Exception as e:
        logger.error(f"ImageToTextPipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=http_error("ImageToTextPipeline error"),
        )
