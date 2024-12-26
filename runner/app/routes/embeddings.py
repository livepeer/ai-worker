# app/routes/embeddings.py
import logging
import os
from typing import Union, List
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import HTTPError, http_error

router = APIRouter()
logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Text to embed")
    model: str = Field("", description="Model to use")
    instruction: Optional[str] = Field(
        None, description="Instruction for instructor models")
    normalize: bool = Field(True, description="Whether to normalize embeddings")


class EmbeddingResponse(BaseModel):
    object: str
    data: List[Dict[str, Union[List[float], int]]]
    model: str
    usage: Dict[str, int]


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
