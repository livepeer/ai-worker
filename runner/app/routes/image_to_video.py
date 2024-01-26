from fastapi import Depends, APIRouter, UploadFile, File, Form
from app.pipelines import ImageToVideoPipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, VideoResponse
import PIL
from typing import Annotated

router = APIRouter()


# TODO: Make model_id optional once Go codegen tool supports OAPI 3.1
# https://github.com/deepmap/oapi-codegen/issues/373
@router.post("/image-to-video", response_model=VideoResponse)
@router.post("/image-to-video/", response_model=VideoResponse, include_in_schema=False)
async def image_to_video(
    image: Annotated[UploadFile, File()],
    model_id: Annotated[str, Form()] = "",
    height: Annotated[int, Form()] = 576,
    width: Annotated[int, Form()] = 1024,
    fps: Annotated[int, Form()] = 7,
    motion_bucket_id: Annotated[int, Form()] = 127,
    noise_aug_strength: Annotated[float, Form()] = 0.02,
    seed: Annotated[int, Form()] = None,
    pipeline: ImageToVideoPipeline = Depends(get_pipeline),
):
    if model_id != "" and model_id != pipeline.model_id:
        raise Exception(
            f"pipeline configured with {pipeline.model_id} but called with {model_id}"
        )

    batch_frames = pipeline(
        PIL.Image.open(image.file).convert("RGB"),
        height=height,
        width=width,
        fps=fps,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        seed=seed,
    )

    output_frames = []
    for frames in batch_frames:
        output_frames.append([{"url": image_to_data_url(frame)} for frame in frames])

    return {"frames": output_frames}
