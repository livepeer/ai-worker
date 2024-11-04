import logging
import os
import shutil
import tempfile
from typing import List, Optional, Tuple

from fastapi import UploadFile
import torch
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_torch_device, get_model_dir
from PIL import ImageFile

from app.utils.errors import InferenceError

ImageFile.LOAD_TRUNCATED_IMAGES = True
from sam2.sam2_video_predictor import SAM2VideoPredictor
import subprocess

logger = logging.getLogger(__name__)


class SegmentAnything2VideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        if torch_device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif torch_device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
        
        self.tm = SAM2VideoPredictor.from_pretrained(
            model_id=model_id,
            device=torch_device,
            **kwargs
        )

    def __call__(
        self, media_file: UploadFile, media_type: str, **kwargs
    ) -> Tuple[List[UploadFile], str, List[Optional[bool]]]:
        try:
            media_file.file.seek(0)

            # Verify that the file isn't empty before proceeding
            file_size = os.fstat(media_file.file.fileno()).st_size
            if file_size == 0:
                raise InferenceError("Uploaded video file is empty")
            else:
                print(f"Video file size: {file_size} bytes")

            temp_dir = tempfile.mkdtemp()
            # TODO: Fix the file type dependency, try passing to ffmpeg without saving to file
            video_path = f"{temp_dir}/input.mp4"
            with open(video_path, "wb") as video_file:
                video_file.write(media_file.file.read())

            # Check if the file was saved properly
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found at {video_path}")
            print(f"Video file saved to {video_path}, size: {os.path.getsize(video_path)} bytes")

            # Run ffmpeg command to extract frames from video
            frame_dir = tempfile.mkdtemp()
            ffmpeg_command = f"ffmpeg -i {video_path} -q:v 2 -start_number 0 {frame_dir}/'%05d.jpg'"
            subprocess.run(ffmpeg_command, shell=True, check=True)
            shutil.rmtree(temp_dir) 

            
            inference_state = self.tm.init_state(video_path=frame_dir)
            shutil.rmtree(frame_dir)

            # TODO: Loop through the points and labels to generate object ids?
            _, _, _ = self.tm.add_new_points_or_box(
                inference_state,
                frame_idx=kwargs.get('frame_idx', None),
                obj_id=1, #TODO: obj_id is hardcoded to 1, should support multiple objects
                points=kwargs.get('points', None),
                labels=kwargs.get('labels', None),
            )
            
            return self.tm.propagate_in_video(inference_state)
    
        except Exception as e:
            raise InferenceError(original_exception=e)

    def __str__(self) -> str:
        return f"Segment Anything 2 Video model_id={self.model_id}"
