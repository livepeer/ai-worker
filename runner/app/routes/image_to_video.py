from fastapi import Depends, APIRouter, UploadFile, File
from app.pipelines import ImageToVideoPipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, VideoResponse
import PIL
from typing import Annotated

router = APIRouter()


@router.post("/image-to-video", response_model=VideoResponse)
@router.post("/image-to-video/", response_model=VideoResponse, include_in_schema=False)
async def image_to_video(
    image: Annotated[UploadFile, File()],
    pipeline: ImageToVideoPipeline = Depends(get_pipeline),
):
    batch_frames = pipeline(PIL.Image.open(image.file).convert("RGB"))

    output_frames = []
    for frames in batch_frames:
        output_frames.append([{"url": image_to_data_url(frame)} for frame in frames])

    return {"frames": output_frames}
