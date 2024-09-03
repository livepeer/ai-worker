import logging
from typing import List, Optional, Tuple

import PIL
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_torch_device, get_model_dir
from app.routes.util import InferenceError
from PIL import ImageFile
from sam2.sam2_image_predictor import SAM2ImagePredictor

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class SegmentAnything2Pipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()

        self.tm = SAM2ImagePredictor.from_pretrained(
            model_id=model_id,
            device=torch_device,
            **kwargs,
        )

    def __call__(
        self, image: PIL.Image, **kwargs
    ) -> Tuple[List[PIL.Image], List[Optional[bool]]]:
        try:
            self.tm.set_image(image)
            prediction = self.tm.predict(**kwargs)
        except Exception as e:
            raise InferenceError(original_exception=e)

        return prediction

    def __str__(self) -> str:
        return f"Segment Anything 2 model_id={self.model_id}"
