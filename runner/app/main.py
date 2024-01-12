from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
import logging
from typing import Dict, Type
from app.pipelines import (
    Pipeline,
    TextToImagePipeline,
    ImageToImagePipeline,
    ImageToVideoPipeline,
    FrameInterpolationPipeline,
    UpscalePipeline,
)

ALLOWED_PIPELINES: Dict[str, Type[Pipeline]] = {
    "text-to-image": TextToImagePipeline,
    "image-to-image": ImageToImagePipeline,
    "image-to-video": ImageToVideoPipeline,
    "frame-interpolation": FrameInterpolationPipeline,
    "upscale": UpscalePipeline,
}

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_logging()
    app.pipeline = get_pipeline()
    logger.info(f"Started up with pipeline {app.pipeline}")
    yield
    logger.info("Shutting down")


def get_pipeline() -> Pipeline:
    pipeline = os.environ["PIPELINE"]
    model_id = os.environ["MODEL_ID"]
    if pipeline not in ALLOWED_PIPELINES:
        raise EnvironmentError(
            f"{pipeline} is not a valid pipeline for model {model_id}"
        )
    # Return initialized pipeline
    return ALLOWED_PIPELINES[pipeline](model_id)


def config_logging():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True,
    )


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}
