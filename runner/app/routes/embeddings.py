# app/routes/embeddings.py
import logging
import os
from typing import Union, List
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import HTTPError, http_error
from app.routes.utils import EmbeddingRequest, EmbeddingResponse

router = APIRouter()
logger = logging.getLogger(__name__)

RESPONSES = {
    status.HTTP_200_OK: {"model": EmbeddingResponse},
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}


@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    responses=RESPONSES,
    operation_id="createEmbeddings",
    description="Generate embeddings for provided text",
    summary="Create Embeddings",
    tags=["embeddings"],
)
async def create_embeddings(
    request: EmbeddingRequest,
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    # Auth check
    auth_token = os.environ.get("AUTH_TOKEN")
    if auth_token:
        if not token or token.credentials != auth_token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
                content=http_error("Invalid bearer token"),
            )

    # Model check
    if request.model != "" and request.model != pipeline.model_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"Pipeline configured with {pipeline.model_id} but called with {request.model}"
            ),
        )

    try:
        response = await pipeline(
            texts=request.input,
            normalize=request.normalize,
            instruction=request.instruction
        )
        return response

    except Exception as e:
        logger.error(f"Embedding generation error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error during embedding generation."}
        )
