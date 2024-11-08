# app/live/pipelines/sam2.py
import io
import logging
import threading
from typing import List, Optional
# from typing import Optional, List, Dict, Literal
import cv2
import numpy as np
from PIL import Image, ImageFile
from pydantic import BaseModel
from Sam2Wrapper import Sam2Wrapper

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

        self.pipe: Optional[Sam2Wrapper] = None
        self.first_frame = True
        self.update_params(**params)

    def update_params(self, **params):
        new_params = Sam2LiveParams(**params)
        self.params = new_params

        logging.info(f"Resetting diffuser for params change")
        self.pipe = Sam2Wrapper(
            model_id_or_path=self.params.model_id,
            point_coords=self.params.point_coords,
            point_labels=self.params.point_labels,
            # Add additional as needed
        )

        self.params = new_params
        self.first_frame = True

    def process_frame(self, frame: Image.Image, **params):
        frame_lock = threading.Lock()

        # Update parameters if provided
        if params:
            self.update_params(**params)

        # Only perform initialization on the first frame
        if self.first_frame:
            width, height = frame.size
            self.pipe.predictor.load_first_frame(frame)
            obj_id = 1
            frame_idx = 0

            # Define the point prompt at one-third from the right, centered vertically
            # this pretty reliably selects the background, for demo purposes
            point = [int(width * 2 / 3), int(height / 2)]
            points = [point]
            labels = [1]

            _, out_obj_ids, out_mask_logits = self.pipe.predictor.add_new_prompt(frame_idx, obj_id, points=points, labels=labels)
        else:
            # Track the object in subsequent frames
            out_obj_ids, out_mask_logits = self.pipe.predictor.track(frame)

        self.first_frame = False
        # Process output mask only if it's non-empty
        if out_mask_logits.shape[0] > 0:
            mask = (out_mask_logits[0, 0] > 0).cpu().numpy().astype("uint8") * 255
        else:
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype="uint8")

        # Convert PIL Image to numpy array
        frame_np = np.array(frame)

        # Resize the mask to match the frame size
        mask_resized = cv2.resize(mask, (frame_np.shape[1], frame_np.shape[0]))

        # Ensure the mask has the same number of channels as the frame
        if len(mask_resized.shape) == 2:
            mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

        # Thread-safe update of the shared `processed_frame` variable
        frame = Image.new("RGB", frame.size, (255, 255, 255)).tobytes()
        with frame_lock:
            processed_frame = mask_resized
            if processed_frame is not None:
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
        
        # Convert the processed frame back to a PIL image
        processed_frame_pil = Image.open(io.BytesIO(frame))
        return processed_frame_pil

        # Create a blank image with the same size as the input frame
        # blank_image = Image.new("RGB", frame.size, (255, 255, 255))
        # return blank_image