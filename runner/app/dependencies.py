from app.pipelines.base import Pipeline
from fastapi import Request


def get_pipeline(request: Request) -> Pipeline:
    return request.app.pipeline
