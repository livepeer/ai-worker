import logging
import os
from typing import Annotated

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.util import HTTPError, AudioResponse, http_error
from fastapi import APIRouter, Depends, Form, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

router = APIRouter()

logger = logging.getLogger(__name__)

RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}

class TextToAudioRequest(BaseModel):
    prompt: str
    seconds_total: int = 30
    steps: int = 100
    cfg_scale: float = 7.0

def handle_pipeline_error(e: Exception) -> JSONResponse:
    """Handles exceptions raised during audio generation."""
    logger.error(f"Audio generation error: {str(e)}")
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_message = "Internal server error during audio generation."

    return JSONResponse(
        status_code=status_code,
        content=http_error(error_message),
    )

@router.post("/text-to-audio", response_model=AudioResponse, responses=RESPONSES)
@router.post(
    "/text-to-audio/",
    response_model=AudioResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def text_to_audio(
    request: TextToAudioRequest,
    model_id: Annotated[str, Form()] = "",
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
        return pipeline(
            prompt=request.prompt,
            seconds_total=request.seconds_total,
            steps=request.steps,
            cfg_scale=request.cfg_scale
        )
    except Exception as e:
        return handle_pipeline_error(e)