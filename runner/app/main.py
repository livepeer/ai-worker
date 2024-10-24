import logging
import os
from contextlib import asynccontextmanager

from app.routes import health
from fastapi import FastAPI
from fastapi.routing import APIRoute

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_logging()

    app.include_router(health.router)

    pipeline = os.environ["PIPELINE"]
    model_id = os.environ["MODEL_ID"]

    app.pipeline = load_pipeline(pipeline, model_id)
    app.include_router(load_route(pipeline))

    logger.info(f"Started up with pipeline {app.pipeline}")
    yield
    logger.info("Shutting down")


def load_pipeline(pipeline: str, model_id: str) -> any:
    match pipeline:
        case "text-to-image":
            from app.pipelines.text_to_image import TextToImagePipeline

            return TextToImagePipeline(model_id)
        case "image-to-image":
            from app.pipelines.image_to_image import ImageToImagePipeline

            return ImageToImagePipeline(model_id)
        case "image-to-video":
            from app.pipelines.image_to_video import ImageToVideoPipeline

            return ImageToVideoPipeline(model_id)
        case "audio-to-text":
            from app.pipelines.audio_to_text import AudioToTextPipeline

            return AudioToTextPipeline(model_id)
        case "frame-interpolation":
            raise NotImplementedError("frame-interpolation pipeline not implemented")
        case "upscale":
            from app.pipelines.upscale import UpscalePipeline

            return UpscalePipeline(model_id)
        case "segment-anything-2":
            from app.pipelines.segment_anything_2 import SegmentAnything2Pipeline

            return SegmentAnything2Pipeline(model_id)
        case "llm":
            from app.pipelines.llm import LLMPipeline

            return LLMPipeline(model_id)
        case "image-to-text":
            from app.pipelines.image_to_text import ImageToTextPipeline

            return ImageToTextPipeline(model_id)
        case "sketch-to-image":
            from app.pipelines.sketch_to_image import SketchToImagePipeline

            return SketchToImagePipeline(model_id)
        case _:
            raise EnvironmentError(
                f"{pipeline} is not a valid pipeline for model {model_id}"
            )


def load_route(pipeline: str) -> any:
    match pipeline:
        case "text-to-image":
            from app.routes import text_to_image

            return text_to_image.router
        case "image-to-image":
            from app.routes import image_to_image

            return image_to_image.router
        case "image-to-video":
            from app.routes import image_to_video

            return image_to_video.router
        case "audio-to-text":
            from app.routes import audio_to_text

            return audio_to_text.router
        case "frame-interpolation":
            raise NotImplementedError("frame-interpolation pipeline not implemented")
        case "upscale":
            from app.routes import upscale

            return upscale.router
        case "segment-anything-2":
            from app.routes import segment_anything_2

            return segment_anything_2.router
        case "llm":
            from app.routes import llm

            return llm.router
        case "image-to-text":
            from app.routes import image_to_text

            return image_to_text.router
        case "sketch-to-image":
            from app.routes import sketch_to_image

            return sketch_to_image.router
        case _:
            raise EnvironmentError(f"{pipeline} is not a valid pipeline")


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
