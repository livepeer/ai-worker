import argparse
import copy
import json

import yaml
from app.main import app, use_route_names_as_operation_ids
from app.routes import (
    audio_to_text,
    health,
    image_to_image,
    image_to_video,
    segment_anything_2,
    text_to_image,
    upscale,
)
from fastapi.openapi.utils import get_openapi
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Specify Endpoints for OpenAPI schema generation.
SERVERS = [
    {
        "url": "https://dream-gateway.livepeer.cloud",
        "description": "Livepeer Cloud Community Gateway",
    },
    {
        "url": "https://livepeer.studio/api/beta/generate",
        "description": "Livepeer Studio Gateway",
    },
]


def get_latest_git_release_tag() -> str:
    """
    Get the latest Git release tag that follows semantic versioning.

    Returns:
        The latest Git release tag, or None if an error occurred.
    """
    try:
        command = (
            "git tag -l 'v*' | grep -E '^v[0-9]+\\.[0-9]+\\.[0-9]+$' | sort -V | "
            "tail -n 1"
        )
        latest_tag = subprocess.check_output(command, shell=True, text=True)
        return latest_tag.strip()
    except subprocess.CalledProcessError as e:
        logger.error("Error occurred while getting the latest git tag: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return None


def translate_to_gateway(openapi: dict) -> dict:
    """Translate the OpenAPI schema from the 'runner' entrypoint to the 'gateway'
    entrypoint created by the https://github.com/livepeer/go-livepeer package.

    .. note::
        Differences between 'runner' and 'gateway' entrypoints:
        - 'health' endpoint is removed.
        - 'model_id' is enforced in all endpoints.
        - 'VideoResponse' schema is updated to match the Gateway's transcoded mp4
            response.

    Args:
        openapi: The OpenAPI schema to be translated.

    Returns:
        The translated OpenAPI schema.
    """
    # Remove 'health' related endpoints and schemas.
    openapi["paths"].pop("/health")
    openapi["components"]["schemas"].pop("HealthCheck")

    # Enforce 'model_id' in all endpoints
    for _, methods in openapi["paths"].items():
        for _, details in methods.items():
            if "requestBody" in details:
                for _, content_details in details["requestBody"]["content"].items():
                    if (
                        "schema" in content_details
                        and "$ref" in content_details["schema"]
                    ):
                        ref = content_details["schema"]["$ref"]
                        schema_name = ref.split("/")[-1]
                        schema = openapi["components"]["schemas"][schema_name]
                        if "model_id" in schema["properties"]:
                            schema["required"].append("model_id")

    # Update the 'VideoResponse' schema to match the Gateway's response.
    # NOTE: This is necessary because the Gateway transcodes the runner's response and
    # returns an mp4 file.
    openapi["components"]["schemas"]["VideoResponse"] = copy.deepcopy(
        openapi["components"]["schemas"]["ImageResponse"]
    )
    openapi["components"]["schemas"]["VideoResponse"]["title"] = "VideoResponse"

    return openapi


def write_openapi(fname: str, entrypoint: str = "runner", version: str = "0.0.0"):
    """Write OpenAPI schema to file.

    Args:
        fname: The file name to write to. The file extension determines the file
            type. Either 'json' or 'yaml'.
        entrypoint: The entrypoint to generate the OpenAPI schema for, either
            'gateway' or 'runner'. Default is 'runner'.
        version: The version to set in the OpenAPI schema. Default is '0.0.0'.
    """
    app.include_router(health.router)
    app.include_router(text_to_image.router)
    app.include_router(image_to_image.router)
    app.include_router(image_to_video.router)
    app.include_router(upscale.router)
    app.include_router(audio_to_text.router)
    app.include_router(segment_anything_2.router)

    use_route_names_as_operation_ids(app)

    logger.info(f"Generating OpenAPI schema for '{entrypoint}' entrypoint...")
    openapi = get_openapi(
        title="Livepeer AI Runner",
        version=version,
        openapi_version=app.openapi_version,
        description="An application to run AI pipelines",
        routes=app.routes,
        servers=SERVERS,
    )

    # Translate OpenAPI schema to 'gateway' side entrypoint if requested.
    if entrypoint == "gateway":
        logger.info(
            "Translating OpenAPI schema from 'runner' to 'gateway' entrypoint..."
        )
        openapi = translate_to_gateway(openapi)
        fname = f"gateway.{fname}"

    # Write OpenAPI schema to file.
    with open(fname, "w") as f:
        logger.info(f"Writing OpenAPI schema to '{fname}'...")
        if fname.endswith(".yaml"):
            yaml.dump(
                openapi,
                f,
                sort_keys=False,
            )
        else:
            json.dump(
                openapi,
                f,
                indent=4,  # Make human readable.
            )
        logger.info("OpenAPI schema generated and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        choices=["json", "yaml"],
        default="yaml",
        help="File type to write to, either 'json' or 'yaml'. Default is 'yaml'",
    )
    parser.add_argument(
        "--entrypoint",
        type=str,
        choices=["runner", "gateway"],
        default=["runner", "gateway"],
        nargs="+",
        help=(
            "The entrypoint to generate the OpenAPI schema for, options are 'runner' "
            "and 'gateway'. Default is both."
        ),
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="The OpenAPI schema version. Default is latest Git semver tag.",
    )
    args = parser.parse_args()

    # Set the 'version' to the latest Git release tag.
    latest_tag = args.version if args.version else get_latest_git_release_tag()

    # Generate orchestrator and Gateway facing OpenAPI schemas.
    logger.info("Generating OpenAPI schema version: $latest_tag")
    for entrypoint in args.entrypoint:
        write_openapi(
            f"openapi.{args.type.lower()}", entrypoint=entrypoint, version=latest_tag
        )
