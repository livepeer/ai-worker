from fastapi import Depends, APIRouter, UploadFile, File, Form
from app.pipelines import ImageToImagePipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, ImageResponse
import PIL
from typing import Annotated

router = APIRouter()


@router.post("/image-to-image", response_model=ImageResponse)
async def image_to_image(
    prompt: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
    pipeline: ImageToImagePipeline = Depends(get_pipeline),
):
    images = pipeline(prompt, PIL.Image.open(image.file))

    output_images = []
    for img in images:
        output_images.append({"url": image_to_data_url(img)})

    return {"images": output_images}
