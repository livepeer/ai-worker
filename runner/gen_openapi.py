from fastapi.openapi.utils import get_openapi
from app.main import app, use_route_names_as_operation_ids
from app.routes import health, text_to_image, image_to_image, image_to_video
import json


def write_openapi_json(fname):
    app.include_router(health.router)
    app.include_router(text_to_image.router)
    app.include_router(image_to_image.router)
    app.include_router(image_to_video.router)

    use_route_names_as_operation_ids(app)

    with open(fname, "w") as f:
        json.dump(
            get_openapi(
                title="Livepeer AI Runner",
                version="0.1.0",
                openapi_version=app.openapi_version,
                description="An application to run AI pipelines",
                routes=app.routes,
            ),
            f,
        )


if __name__ == "__main__":
    write_openapi_json("openapi.json")
