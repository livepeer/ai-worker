from fastapi import FastAPI
from fastapi.routing import APIRoute
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

    use_route_names_as_operation_ids(app)

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
            from app.pipelines import ImageToImagePipeline
            from app.routes import image_to_image

            config["pipeline"] = ImageToImagePipeline(model_id)
            config["route"] = image_to_image.router
        case "image-to-video":
            from app.pipelines import ImageToVideoPipeline
            from app.routes import image_to_video

            config["pipeline"] = ImageToVideoPipeline(model_id)
            config["route"] = image_to_video.router
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


def use_route_names_as_operation_ids(app: FastAPI) -> None:
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name


app = FastAPI(lifespan=lifespan)
