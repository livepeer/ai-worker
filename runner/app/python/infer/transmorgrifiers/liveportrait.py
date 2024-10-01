from typing import Literal, Optional, List, Dict
from PIL import Image
from pydantic import BaseModel, Field

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

from .interface import Transmorgrifier


def make_flip_path(rel_path):
    return os.path.normpath(os.path.join(base_flip_dir, rel_path))

class LivePortraitParams(BaseModel):
    src_image: str = 'flame-smile'
    animal: bool = False
    cfg: str = 'configs/trt_infer.yaml'

class LivePortrait(Transmorgrifier):
    def __init__(self, **params):
        super().__init__(**params)
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
        logging.info(f"params: {params}")
        self.params = LivePortraitParams(**params)
        self.params.cfg = make_flip_path(self.params.cfg)
        self.params.src_image = make_flip_path(f"assets/examples/source/{self.params.src_image}.jpg")

        logging.info(f"params.cfg: {self.params}")

        self.infer_cfg = OmegaConf.load(self.params.cfg)
        self.infer_cfg.infer_params.mask_crop_path = make_flip_path(self.infer_cfg.infer_params.mask_crop_path)
        for model_name in self.infer_cfg.models:
            model_params = self.infer_cfg.models[model_name]
            if isinstance(model_params.model_path, str):
                model_params.model_path = make_flip_path(model_params.model_path)
            else:
                model_params.model_path = [make_flip_path(path) for path in model_params.model_path]

        self.pipe: Optional[FasterLivePortraitPipeline] = FasterLivePortraitPipeline(cfg=self.infer_cfg, is_animal=self.params.animal)
        self.first_frame = True

        prepared_src = self.pipe.prepare_source(self.params.src_image)
        if not prepared_src:
            raise ValueError(f"no face in {self.params.src_image}!")
        else:
            self.prepared_src = prepared_src
