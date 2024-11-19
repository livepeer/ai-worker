import io
import logging
import threading
import time
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel
from Sam2Wrapper import Sam2Wrapper
from .interface import Pipeline

logger = logging.getLogger(__name__)

class Sam2LiveParams(BaseModel):
    class Config:
        extra = "forbid"

    model_id: str = "facebook/sam2-hiera-tiny"
    point_coords: Optional[List[List[int]]] = [[1,1]]
    point_labels: Optional[List[int]] = [1]
    obj_ids: Optional[List[int]] = [1]
    show_point: bool = False
    show_overlay: bool = True

    def __init__(self, **data): 
        super().__init__(**data)

class Sam2Live(Pipeline):
    def __init__(self, **params):
        super().__init__(**params)
        self.pipe: Optional[Sam2Wrapper] = None
        self.first_frame = True
        self.update_params(**params)

    def update_params(self, **params):
        # Only update point coordinates and labels if both are provided
        if ('point_coords' in params) != ('point_labels' in params):
            raise ValueError("Both point_coords and point_labels must be updated together")
            
        # Preserve existing values if neither is provided
        if hasattr(self, 'params'):
            if 'point_coords' not in params and hasattr(self.params, 'point_coords'):
                params["point_coords"] = self.params.point_coords
                
            if 'point_labels' not in params and hasattr(self.params, 'point_labels'):
                params["point_labels"] = self.params.point_labels
            
        new_params = Sam2LiveParams(**params)
        self.params = new_params
        
        #TODO: Only reload the model if the point, label coordinates or model has changed
        self.first_frame = True

        logging.info(f"Setting parameters for sam2")
        self.pipe = Sam2Wrapper(
            model_id_or_path=self.params.model_id,
            point_coords=self.params.point_coords,
            point_labels=self.params.point_labels,
            obj_ids=self.params.obj_ids,
            show_point=self.params.show_point,
            show_overlay=self.params.show_overlay
            
            # Add additional as needed
        )


    def _process_mask(self, mask: np.ndarray, frame_shape: tuple, color: list[list[int]]) -> np.ndarray:
        """Process and resize mask if needed."""
        logger.info(f"Mask input shape: {mask.shape}")
        if mask.shape[0] == 0:
            logger.warning("Empty mask received")
            return np.zeros((frame_shape[0], frame_shape[1]), dtype="uint8")
        
        colors = [
            [255, 0, 255],  # Purple
            [0, 255, 255],  # Yellow
            [255, 255, 0],  # Cyan
            [0, 255, 0],    # Green
            [255, 0, 0],    # Blue
        ]
        
        # Initialize the combined colored mask with alpha channel
        combined_colored_mask = np.zeros((frame_shape[0], frame_shape[1], 4), dtype="uint8")
        
        # Process each mask
        for i in range(mask.shape[0]):
            current_mask = (mask[i, 0] > 0).cpu().numpy().astype("uint8") * 255
            if current_mask.shape[:2] != frame_shape[:2]:
                current_mask = cv2.resize(current_mask, (frame_shape[1], frame_shape[0]))
            
            # Create BGRA mask with transparency
            colored_mask = np.zeros((frame_shape[0], frame_shape[1], 4), dtype="uint8")
            color = colors[i % len(colors)]
            colored_mask[current_mask > 0] = color + [128]  # Add alpha value of 128
            
            # Alpha blend with existing masks
            alpha = colored_mask[:, :, 3:4] / 255.0
            combined_colored_mask = (1 - alpha) * combined_colored_mask + alpha * colored_mask

        # Convert back to BGR for display
        combined_colored_mask = combined_colored_mask[:, :, :3].astype("uint8")
        return combined_colored_mask

    def process_frame(self, frame: Image.Image, **params) -> Image.Image:
        frame_lock = threading.Lock()
        try:
            if params:
                self.update_params(**params)

            # Convert image formats
            frame_array = np.array(frame)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)
            if self.first_frame:
                self.pipe.predictor.load_first_frame(frame)
                
                # For each obj_id, add a new point and label
                for idx, obj_id in enumerate(self.params.obj_ids):
                    if self.params.point_coords and self.params.point_labels:
                        point = self.params.point_coords[idx]
                        label = self.params.point_labels[idx]
                        _, _, mask_logits = self.pipe.predictor.add_new_prompt(
                            frame_idx=0, 
                            obj_id=obj_id, 
                            points=[point], 
                            labels=[label]
                        )
                
                # logger.info(f"First frame mask_logits shape: {mask_logits.shape}")
                self.first_frame = False
            else:
                out_obj_ids, mask_logits = self.pipe.predictor.track(frame)
                # logger.info(f"Tracking mask_logits shape: {mask_logits.shape}")
            
            # Initialize overlay with original frame
            overlay = frame_bgr.copy()
            
            # Only apply mask overlay if show_overlay is True
            if self.params.show_overlay:
                # Loop through each object ID and apply the corresponding mask
                for i, obj_id in enumerate(out_obj_ids):
                    logger.info(f"Processing mask for object ID: {obj_id}")
                    single_mask_logits = mask_logits[i:i+1]
                    colors = [
                        [255, 0, 255],  # Purple
                        [0, 255, 255],  # Yellow
                        [255, 255, 0],  # Cyan
                        [0, 255, 0],    # Green
                        [255, 0, 0],    # Blue
                    ]
                    
                    colored_mask = self._process_mask(single_mask_logits, frame_bgr.shape, colors[i])
                    overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

            # Draw points on the overlay if needed
            if hasattr(self.params, 'show_point') and self.params.show_point:
                for point in self.params.point_coords:
                    cv2.circle(overlay, tuple(point), radius=5, color=(0, 0, 255), thickness=-1)

            # Convert back to PIL Image
            with frame_lock:
                _, buffer = cv2.imencode('.jpg', overlay)
                result = Image.open(io.BytesIO(buffer.tobytes()))

            return result

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return Image.new("RGB", frame.size, (255, 255, 255))