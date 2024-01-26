from fastapi import Depends, APIRouter, UploadFile, File, Form
from app.pipelines import ImageToImagePipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, ImageResponse
import PIL
from typing import Annotated

router = APIRouter()


# TODO: Make model_id optional once Go codegen tool supports OAPI 3.1
# https://github.com/deepmap/oapi-codegen/issues/373
@router.post("/image-to-image", response_model=ImageResponse)
@router.post("/image-to-image/", response_model=ImageResponse, include_in_schema=False)
async def image_to_image(
    prompt: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
    model_id: Annotated[str, Form()] = "",
    strength: Annotated[float, Form()] = 0.8,
    guidance_scale: Annotated[float, Form()] = 7.5,
    negative_prompt: Annotated[str, Form()] = "",
    seed: Annotated[int, Form()] = None,
    pipeline: ImageToImagePipeline = Depends(get_pipeline),
):
    if model_id != "" and model_id != pipeline.model_id:
        raise Exception(
            f"pipeline configured with {pipeline.model_id} but called with {model_id}"
        )

    images = pipeline(
        prompt,
        PIL.Image.open(image.file).convert("RGB"),
        strength=strength,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        seed=seed,
    )

    output_images = []
    for img in images:
        output_images.append({"url": image_to_data_url(img)})

    return {"images": output_images}
