from fastapi.openapi.utils import get_openapi
from app.main import app, use_route_names_as_operation_ids
from app.routes import health, text_to_image, image_to_image, image_to_video
import json
import yaml
import argparse


def write_openapi(fname):
    """Write OpenAPI schema to file.

    Args:
        fname (str): The file name to write to. The file extension determines the file
            type. Either 'json' or 'yaml'.
    """
    app.include_router(health.router)
    app.include_router(text_to_image.router)
    app.include_router(image_to_image.router)
    app.include_router(image_to_video.router)

    use_route_names_as_operation_ids(app)

    # Write OpenAPI schema to file.
    with open(fname, "w") as f:
        print(f"Writing OpenAPI schema to '{fname}'...")
        if fname.endswith(".yaml"):
            yaml.dump(
                get_openapi(
                    title="Livepeer AI Runner",
                    version="0.1.0",
                    openapi_version=app.openapi_version,
                    description="An application to run AI pipelines",
                    routes=app.routes,
                ),
                f,
                sort_keys=False,
            )
        else:
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
        print("OpenAPI schema generated and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        choices=["json", "yaml"],
        default="json",
        help="File type to write to, either 'json' or 'yaml'. Default is 'json'",
    )
    args = parser.parse_args()

    write_openapi(f"openapi.{args.type.lower()}")
