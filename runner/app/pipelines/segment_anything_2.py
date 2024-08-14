import logging
from typing import List, Optional, Tuple

import PIL
from app.pipelines.base import Pipeline
from app.pipelines.utils import (
    get_model_dir,
    get_torch_device,
)
from sam2.sam2_image_predictor import SAM2ImagePredictor

logger = logging.getLogger(__name__)


class SegmentAnything2Pipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id

        torch_device = get_torch_device()

        self.tm = SAM2ImagePredictor.from_pretrained(
            model_id=model_id,
            device=torch_device,
        )

    def __call__(
        self, image: PIL.Image, **kwargs
    ) -> Tuple[List[PIL.Image], List[Optional[bool]]]:

        self.tm.set_image(image)

        return self.tm.predict(**kwargs)

    def __str__(self) -> str:
        return f"Segment Anything 2 model_id={self.model_id}"
