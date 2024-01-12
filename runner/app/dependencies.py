from fastapi import Request
from app.pipelines import Pipeline


def get_pipeline(request: Request) -> Pipeline:
    return request.app.pipeline
