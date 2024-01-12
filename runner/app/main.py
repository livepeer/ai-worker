from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
import logging
from typing import Any
from pydantic import BaseModel


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_logging()

    config = load_pipeline()
    app.pipeline = config["pipeline"]
    app.include_router(config["route"])

    logger.info(f"Started up with pipeline {app.pipeline}")
    yield
    logger.info("Shutting down")


class PipelineConfig(BaseModel):
    pipeline: Any
    route: Any


def load_pipeline() -> PipelineConfig:
    pipeline = os.environ["PIPELINE"]
    model_id = os.environ["MODEL_ID"]

    config = {}
    match pipeline:
        case "text-to-image":
            from app.pipelines import TextToImagePipeline
            from app.routes import text_to_image

            config["pipeline"] = TextToImagePipeline(model_id)
            config["route"] = text_to_image.router
        case "image-to-image":
            raise NotImplementedError("image-to-image pipeline not implemented")
        case "image-to-video":
            raise NotImplementedError("image-to-video pipeline not implemented")
        case "frame-interpolation":
            raise NotImplementedError("frame-interpolation pipeline not implemented")
        case "upscale":
            raise NotImplementedError("upscale pipeline not implemented")
        case _:
            raise EnvironmentError(
                f"{pipeline} is not a valid pipeline for model {model_id}"
            )

    return config


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
