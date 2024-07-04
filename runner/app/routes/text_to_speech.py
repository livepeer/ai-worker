from typing import Annotated
from fastapi import Depends, APIRouter, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from app.pipelines.base import Pipeline
from app.routes.util import AudioResponse
from app.dependencies import get_pipeline
import logging
import os

class HTTPError(BaseModel):
    detail: str

router = APIRouter()

logger = logging.getLogger(__name__)

responses = {
    400: {"content": {"application/json": {"schema": HTTPError.schema()}}},
    500: {"content": {"application/json": {"schema": HTTPError.schema()}}},
    200: {
        "content": {
            "audio/mp4": {},
        }
    }
}

class TextToSpeechParams(BaseModel):
    text_input: Annotated[str, Form()] = ""
    model_id: str = ""


@router.post("/text-to-speech",
    response_model=AudioResponse,
    responses=responses)
async def text_to_speech(
    params: TextToSpeechParams,
    pipeline: Pipeline = Depends(get_pipeline),
):

    try:
        if not params.text_input:
            raise ValueError("text_input is required and cannot be empty.")
        
        result = pipeline(params.text_input)

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        return JSONResponse(
            status_code=400,
            content={"detail": str(ve)},
        )

    except Exception as e:
        logger.error(f"TextToSpeechPipeline error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal Server Error: {str(e)}"},
        )

    if os.path.exists(result):
        return FileResponse(path=result, media_type='audio/mp4', filename="generated_audio.mp4")
    else:
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"no output found for {result}"
            },
        )
