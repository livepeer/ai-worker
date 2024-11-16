import logging
import os
from typing import Annotated

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import HTTPError, TextSentimentAnalysisResponse, http_error
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import Field

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
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}


def handle_pipeline_error(e: Exception) -> JSONResponse:
    """Handles exceptions raised during text processing.

    Args:
        e: The exception raised during text processing.

    Returns:
        A JSONResponse with the appropriate error message and status code.
    """
    logger.error(f"Text classification processing error: {str(e)}")
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_message = "Internal server error during text processing."

    return JSONResponse(
        status_code=status_code,
        content=http_error(error_message),
    )


@router.post(
    "/text-sentiment-analysis",
    response_model=TextSentimentAnalysisResponse,
    responses=RESPONSES,
    description="Analyze the sentiment of a given text inputs.",
    operation_id="analyzeSentiment",
    summary="Text Sentiment Analysis",
    tags=["analysis"],
    openapi_extra={"x-speakeasy-name-override": "textSentimentAnalysis"},
)
@router.post(
    "/text-sentiment-analysis/",
    response_model=TextSentimentAnalysisResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def text_sentiment_analysis(
    model_id: Annotated[
        str,
        Field(
            default="",
            description="Hugging Face model ID used for text classsification."
        ),
    ],
    text_input: Annotated[
        str,
        Field(description="Text to analyze. For multiple sentences, separate them with commas.")
    ],
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
        scores = pipeline(text=text_input.split(","))
        return {"results": scores}
    except Exception as e:
        return handle_pipeline_error(e)
