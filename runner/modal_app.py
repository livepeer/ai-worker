from modal import Image, Stub, asgi_app, enter, method, Secret, Volume
import logging
from pathlib import Path
from app.main import (
    config_logging,
    load_route,
    use_route_names_as_operation_ids,
)
from app.routes import health

stub = Stub("livepeer-ai-runner")
pipeline_image = (
    Image.from_registry("livepeer/ai-runner:latest")
    .workdir("/app")
    .env({"BFLOAT16": "true"})
)
api_image = Image.debian_slim(python_version="3.11").pip_install(
    "pydantic==2.6.1", "fastapi==0.109.2", "pillow"
)
downloader_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub==0.20.2",
        "hf-transfer==0.1.4",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
)
models_volume = Volume.persisted("models")
models_path = Path("/models")

logger = logging.getLogger(__name__)


@stub.function(
    image=downloader_image, volumes={models_path: models_volume}, timeout=3600
)
def download_model(model_id: str):
    from huggingface_hub import snapshot_download

    try:
        # TODO: Handle case where there are no fp16 safetensors available
        allow_patterns = ["*unet.safetensors", "*.fp16.safetensors", "*.json", "*.txt"]
        ignore_patterns = [".onnx", ".onnx_data"]
        cache_dir = "/models"

        snapshot_download(
            model_id,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        logger.info(f"Downloaded model {model_id} to volume")
        models_volume.commit()
    except Exception:
        logger.exception(f"Failed to download model {model_id} to volume")
        raise


@stub.cls(
    gpu="A10G",
    image=pipeline_image,
    memory=1024,
    volumes={models_path: models_volume},
    container_idle_timeout=3 * 60,
)
class Pipeline:
    def __init__(self, pipeline: str, model_id: str):
        self.pipeline = pipeline
        self.model_id = model_id

    @enter()
    def enter(self):
        from app.main import load_pipeline

        model_dir = "models--" + self.model_id.replace("/", "--")
        path = models_path / model_dir
        if not path.exists():
            models_volume.reload()

        if not path.exists():
            raise Exception(f"No model found at {path}")

        self.pipe = load_pipeline(self.pipeline, self.model_id)

    @method()
    def predict(self, **kwargs):
        return self.pipe(**kwargs)


# Wrap Pipeline for dependency injection in the runner FastAPI route
class RunnerPipeline:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.model_id = pipeline.model_id

    def __call__(self, **kwargs):
        return self.pipeline.predict.remote(**kwargs)


def make_api(pipeline: str, model_id: str):
    from fastapi import FastAPI

    config_logging()

    app = FastAPI()

    app.include_router(health.router)

    app.pipeline = RunnerPipeline(Pipeline(pipeline, model_id))
    app.include_router(load_route(pipeline))

    use_route_names_as_operation_ids(app)

    return app


@stub.function(image=api_image, secrets=[Secret.from_name("api-auth-token")])
@asgi_app()
def text_to_image_sdxl_lightning_api():
    return make_api("text-to-image", "ByteDance/SDXL-Lightning")


@stub.function(image=api_image, secrets=[Secret.from_name("api-auth-token")])
@asgi_app()
def text_to_image_sdxl_turbo_api():
    return make_api("text-to-image", "stabilityai/sdxl-turbo")


@stub.function(image=api_image, secrets=[Secret.from_name("api-auth-token")])
@asgi_app()
def text_to_image_sd_1_5_api():
    return make_api("text-to-image", "runwayml/stable-diffusion-v1-5")


@stub.function(image=api_image, secrets=[Secret.from_name("api-auth-token")])
@asgi_app()
def text_to_image_sdxl_api():
    return make_api("text-to-image", "stabilityai/stable-diffusion-xl-base-1.0")
