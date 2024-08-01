import logging
import os
from typing import Annotated, Optional, List
from fastapi import APIRouter, Depends, Form, status, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.util import HTTPError, LlmResponse, TextResponse, http_error
import json

router = APIRouter()

logger = logging.getLogger(__name__)

RESPONSES = {
    status.HTTP_200_OK: {"model": LlmResponse},
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}


@router.post("/llm-generate",
             response_model=LlmResponse, responses=RESPONSES)
@router.post("/llm-generate/", response_model=LlmResponse, responses=RESPONSES, include_in_schema=False)
async def llm_generate(
    prompt: Annotated[str, Form()],
    model_id: Annotated[str, Form()] = "",
    system_msg: Annotated[str, Form()] = "",
    temperature: Annotated[float, Form()] = 0.7,
    max_tokens: Annotated[int, Form()] = 256,
    history: Annotated[str, Form()] = "[]",  # We'll parse this as JSON
    stream: Annotated[bool, Form()] = False,
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
        history_list = json.loads(history)
        if not isinstance(history_list, list):
            raise ValueError("History must be a JSON array")

        generator = pipeline(
            prompt=prompt,
            history=history_list,
            system_msg=system_msg if system_msg else None,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if stream:
            return StreamingResponse(stream_generator(generator), media_type="text/event-stream")
        else:
            full_response = ""
            async for chunk in generator:
                if isinstance(chunk, dict):
                    tokens_used = chunk["tokens_used"]
                    break
                full_response += chunk

            return LlmResponse(response=full_response, tokens_used=tokens_used)

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Invalid JSON format for history"}
        )
    except ValueError as ve:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(ve)}
        )
    except Exception as e:
        logger.error(f"LLM processing error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error during LLM processing."}
        )


async def stream_generator(generator):
    try:
        async for chunk in generator:
            if isinstance(chunk, dict):  # This is the final result
                yield f"data: {json.dumps(chunk)}\n\n"
                break
            else:
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
