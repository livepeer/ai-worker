import logging
import os
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, Form, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.util import HTTPError, LlmResponse, TextResponse, http_error

router = APIRouter()

logger = logging.getLogger(__name__)

RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}


@router.post("/llm-generate", response_model=LlmResponse, responses=RESPONSES)
@router.post("/llm-generate/", response_model=LlmResponse, responses=RESPONSES, include_in_schema=False)
async def llm_generate(
    prompt: Annotated[str, Form()],
    model_id: Annotated[str, Form()] = "",
    system_msg: Annotated[str, Form()] = None,
    temperature: Annotated[float, Form()] = None,
    max_tokens: Annotated[int, Form()] = None,
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
        result = pipeline(
            prompt=prompt,
            system_msg=system_msg,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"LLM processing error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=http_error("Internal server error during LLM processing."),
        )
