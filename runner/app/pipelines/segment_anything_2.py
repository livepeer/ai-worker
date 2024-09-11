import logging
import os
import shutil
import tempfile
from typing import List, Optional, Tuple

from PIL import Image
from fastapi import UploadFile
import torch
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_torch_device, get_model_dir
from app.routes.util import InferenceError
from PIL import ImageFile
from sam2.sam2_image_predictor import SAM2ImagePredictor

ImageFile.LOAD_TRUNCATED_IMAGES = True
from sam2.sam2_video_predictor import SAM2VideoPredictor
import subprocess

logger = logging.getLogger(__name__)


class SegmentAnything2Pipeline(Pipeline):
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

        self.tm = SAM2ImagePredictor.from_pretrained(
            model_id=model_id,
            device=torch_device,
            **kwargs,
        )
        
        self.tm_vid = SAM2VideoPredictor.from_pretrained(
            model_id=model_id,
            device=torch_device,
        )

    def __call__(
        self, media_file: UploadFile, media_type: str, **kwargs
    ) -> Tuple[List[UploadFile], str, List[Optional[bool]]]:
        if media_type == "image":
            try:
                image = Image.open(media_file.file).convert("RGB")
                self.tm.set_image(image)
                prediction = self.tm.predict(**kwargs)
            except Exception as e:
                raise InferenceError(original_exception=e)
        elif media_type == "video":
            try:
                temp_dir = tempfile.mkdtemp()
                # TODO: Fix the file type dependency, try passing to ffmpeg without saving to file
                video_path = f"{temp_dir}/input.mp4"
                with open(video_path, "wb") as video_file:
                    video_file.write(media_file.file.read())
                
                # Run ffmpeg command to extract frames from video
                frame_dir = tempfile.mkdtemp()
                output_pattern = f"{frame_dir}/%05d.jpg"
                ffmpeg_command = f"ffmpeg -i {video_path} -q:v 2 -start_number 0 {output_pattern}"
                subprocess.run(ffmpeg_command, shell=True, check=True)
                shutil.rmtree(temp_dir) 

                # Limit to the first 500 frames to avoid running out of memory
                frame_files = sorted(
                    [f for f in os.listdir(frame_dir) if f.endswith('.jpg')]
                )
                for frame_file in frame_files[:-500]:
                    os.remove(os.path.join(frame_dir, frame_file))
                
                inference_state = self.tm_vid.init_state(video_path=frame_dir)
                shutil.rmtree(frame_dir)

                _, out_obj_ids, out_mask_logits = self.tm_vid.add_new_points_or_box(
                    inference_state,
                    frame_idx=kwargs.get('frame_idx', None),
                    obj_id=1,
                    points=kwargs.get('points', None),
                    labels=kwargs.get('labels', None),
                    )
                
                for out_frame_idx, out_obj_ids, out_mask_logits in self.tm_vid.propagate_in_video(inference_state):
                   return {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                
            except Exception as e:
                raise InferenceError(original_exception=e)

        return prediction

    def __str__(self) -> str:
        return f"Segment Anything 2 model_id={self.model_id}"
