from PIL import Image
from pydantic import BaseModel
import hashlib

from omegaconf import OmegaConf
import cv2
import numpy as np


import sys
import os
import logging
# FasterLivePotrait modules imports files from the root of the project, so we need to monkey patch the sys path
base_flip_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "FasterLivePortrait"))
sys.path.append(base_flip_dir)
from FasterLivePortrait.src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

from .interface import Pipeline


def make_flip_path(rel_path):
    return os.path.normpath(os.path.join(base_flip_dir, rel_path))

# Subset of configs from the trt_infer.yaml#infer_params
class LivePortraitInferParams(BaseModel):
    class Config:
        extra = 'forbid'

    crop_driving_video: bool = False
    normalize_lip: bool = True
    source_video_eye_retargeting: bool = False
    video_editing_head_rotation: bool = False
    eye_retargeting: bool = False
    lip_retargeting: bool = False
    stitching: bool = True
    relative_motion: bool = True
    pasteback: bool = True
    do_crop: bool = True
    do_rot: bool = True

    lip_normalize_threshold: float = 0.03
    source_video_eye_retargeting_threshold: float = 0.18
    driving_smooth_observation_variance: float = 1e-7
    driving_multiplier: float = 1.0

    def to_omegaconf(self) -> OmegaConf:
        is_flag = lambda field: field not in ['lip_normalize_threshold', 'source_video_eye_retargeting_threshold', 'driving_smooth_observation_variance', 'driving_multiplier']
        params = {
            f'{"flag_" if is_flag(field) else ""}{field}': getattr(self, field)
            for field in self.__dict__
        }
        return OmegaConf.create(params)

class LivePortraitParams(BaseModel):
    class Config:
        extra = 'forbid'

    src_image: str = 'flame-smile'
    animal: bool = False
    infer_params: LivePortraitInferParams = LivePortraitInferParams()

base_pipe_config_path = make_flip_path('configs/trt_infer.yaml')

class LivePortrait(Pipeline):
    def __init__(self, **params):
        super().__init__(**params)
        self.pipe = None
        self.update_params(**params)

    def process_frame(self, image: Image.Image) -> Image.Image:
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        _, out_crop, _ = self.pipe.run(cv2_image, self.pipe.src_imgs[0], self.pipe.src_infos[0], first_frame=self.first_frame)
        self.first_frame = False

        if out_crop is None:
            logging.info(f"No face in driving frame")
            return image

        return Image.fromarray(out_crop)

    def update_params(self, **params):
        new_params = LivePortraitParams(**params)
        if not os.path.isabs(new_params.src_image):
            new_params.src_image = make_flip_path(f"assets/examples/source/{new_params.src_image}.jpg")

        logging.info(f"liveportrait new params: {new_params}")

        new_cfg = OmegaConf.load(base_pipe_config_path)
        new_cfg.infer_params = OmegaConf.merge(new_cfg.infer_params, new_params.infer_params.to_omegaconf())
        new_cfg.infer_params.mask_crop_path = make_flip_path(new_cfg.infer_params.mask_crop_path)
        for model_name in new_cfg.models:
            model_params = new_cfg.models[model_name]
            if isinstance(model_params.model_path, str):
                model_params.model_path = make_flip_path(model_params.model_path)
            else:
                model_params.model_path = [make_flip_path(path) for path in model_params.model_path]

        config_hash = hashlib.md5(str(new_cfg).encode()).hexdigest()
        logging.info(f"liveportrait new config hash: {config_hash}")

        new_pipe = FasterLivePortraitPipeline(cfg=new_cfg, is_animal=new_params.animal)

        prepared_src = new_pipe.prepare_source(new_params.src_image)
        if not prepared_src:
            raise ValueError(f"no face in {new_params.src_image}!")

        if self.pipe is not None:
            self.pipe.clean_models()

        self.params = new_params
        self.cfg = new_cfg
        self.pipe = new_pipe
        self.first_frame = True
        self.prepared_src = prepared_src
