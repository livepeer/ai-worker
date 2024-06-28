import argparse
import json
import os
import copy

import yaml
from app.main import app, use_route_names_as_operation_ids
from app.routes import health, image_to_image, image_to_video, text_to_image, upscale, speech_to_text
from fastapi.openapi.utils import get_openapi

# Specify Endpoints for OpenAPI schema generation.
SERVERS = [
    {
        "url": "https://dream-gateway.livepeer.cloud",
        "description": "Livepeer Cloud Community Gateway",
    },
]


def translate_to_gateway(openapi):
    """Translate the OpenAPI schema from the 'runner' entrypoint to the 'gateway'
    entrypoint created by the https://github.com/livepeer/go-livepeer package.

    .. note::
        Differences between 'runner' and 'gateway' entrypoints:
        - 'health' endpoint is removed.
        - 'model_id' is enforced in all endpoints.
        - 'VideoResponse' schema is updated to match the Gateway's transcoded mp4
            response.

    Args:
        openapi (dict): The OpenAPI schema to be translated.

    Returns:
        dict: The translated OpenAPI schema.
    """
    # Remove 'health' endpoint
    openapi["paths"].pop("/health")

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


def write_openapi(fname, entrypoint="runner"):
    """Write OpenAPI schema to file.

    Args:
        fname (str): The file name to write to. The file extension determines the file
            type. Either 'json' or 'yaml'.
        entrypoint (str): The entrypoint to generate the OpenAPI schema for, either
            'gateway' or 'runner'. Default is 'runner'.
    """
    app.include_router(health.router)
    app.include_router(text_to_image.router)
    app.include_router(image_to_image.router)
    app.include_router(image_to_video.router)
    app.include_router(upscale.router)
    app.include_router(speech_to_text.router)

    use_route_names_as_operation_ids(app)

    print(f"Generating OpenAPI schema for '{entrypoint}' entrypoint...")
    openapi = get_openapi(
        title="Livepeer AI Runner",
        version="0.1.0",
        openapi_version=app.openapi_version,
        description="An application to run AI pipelines",
        routes=app.routes,
        servers=SERVERS,
    )

    # Translate OpenAPI schema to 'gateway' side entrypoint if requested.
    if entrypoint == "gateway":
        print("Translating OpenAPI schema from 'runner' to 'gateway' entrypoint...")
        openapi = translate_to_gateway(openapi)
        fname = os.path.splitext(fname)[0] + "_gateway" + os.path.splitext(fname)[1]

    # Write OpenAPI schema to file.
    with open(fname, "w") as f:
        print(f"Writing OpenAPI schema to '{fname}'...")
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
    parser.add_argument(
        "--entrypoint",
        type=str,
        choices=["gateway", "runner"],
        default="runner",
        help=(
            "The entrypoint to generate the OpenAPI schema for, either 'gateway' or "
            "'runner'. Default is 'runner'",
        ),
    )
    args = parser.parse_args()

    write_openapi(f"openapi.{args.type.lower()}", args.entrypoint)
