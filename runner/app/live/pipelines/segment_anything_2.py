# app/live/pipelines/sam2.py
import logging
from typing import List
# from typing import Optional, List, Dict, Literal
import torch
import PIL
from PIL import Image, ImageFile
from pydantic import BaseModel
from sam2.sam2_image_predictor import SAM2ImagePredictor

# from .segment_anything_2 import Sam2Wrapper  # Adjust the import path as needed

from .interface import Pipeline
logger = logging.getLogger(__name__)

class Sam2LiveParams(BaseModel):
    class Config:
        extra = "forbid"

    model_id: str = "facebook/sam2-hiera-large"
    point_coords: List[List[int]] = [[10, 20], [80, 100]]
    point_labels: List[int] = [5, 5]

    def __init__(self, **data): 
        super().__init__(**data)

class Sam2Live(Pipeline):
    def __init__(self, **params):
        super().__init__(**params)
        self.update_params(**params)
        # self.pipe: Optional[Sam2Wrapper] = None
        self.first_frame = True
        
        # Should be moved as parameter passed to infer.py later
        if torch.cuda.is_available():
            torch_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            torch_device = torch.device("mps")
        else:
            torch_device = torch.device("cpu")

        kwargs = params
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id="facebook/sam2-hiera-large", device=torch_device, **kwargs)

        # self.predictor = SAM2ImagePredictor.from_pretrained(
        #     model_id="facebook/sam2-hiera-large",
        #     device=torch_device,
        #     **kwargs
        # )
        self.current_image = None
        # self.update_params(**params)

    def set_image(self, image: Image.Image):
        self.predictor.set_image(image)
        self.current_image = image

    def predict(self, **kwargs):
        if self.current_image is None:
            raise ValueError("Must call set_image() before predict()")
        return self.predictor.predict(**kwargs)
    
    def update_params(self, **params):
        new_params = Sam2LiveParams(**params)
        self.params = new_params

        logging.info(f"Resetting diffuser for params change")
        # pipe = Sam2Wrapper(
        #     model_id_or_path=self.params.model_id,
        #     point_coords=self.params.point_coords,
        #     point_labels=self.params.point_labels,
        #     # Add additional as needed
        # )

        # This might only be needed when a new image is set, in which case it maybe could be moved to set_image()
        self.params = new_params
        # self.pipe = pipe
        self.first_frame = True

    def process_frame(self, frame: Image.Image, **params):
        # self.set_image(frame)

        # Update parameters if provided
        if params:
            self.update_params(**params)

        # Get prediction results using current parameters
        results = self.predict(
            point_coords=self.point_coords,
            point_labels=self.point_labels
        )

        #follow image predictor example to segment the frame and render the box or mask

        # Create a blank image with the same size as the input frame
        blank_image = Image.new("RGB", frame.size, (255, 255, 255))
        return blank_image