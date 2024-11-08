# Copied from StreamDiffusion/utils/wrapper.py

import gc
import logging
import os
import traceback
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

class Sam2Wrapper:
    def __init__(
        self,
        model_id_or_path: str,
        device: Optional[str] = None,
        **kwargs
    ):
        self.model_id = model_id_or_path
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.predictor = SAM2ImagePredictor.from_pretrained(
            model_id=model_id_or_path,
            device=device,
            **kwargs
        )
  
    def __call__(
        self, 
        image: Image.Image,
        **kwargs
    ) -> Tuple[List[Image.Image], List[Optional[bool]]]:
        try:
            self.predictor.set_image(image)
            prediction = self.predictor.predict(**kwargs)
            return prediction
        except Exception as e:
            traceback.print_exc()
            raise e

    def __str__(self) -> str:
        return f"Sam2Wrapper model_id={self.model_id}"
    

# class Sam2Wrapper:
#     def __init__(
#         self,
#         model_id_or_path: str,
#         device: Optional[str] = None,
#         **kwargs
#     ):
#         self.model_id = model_id_or_path
        
#         if device is None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"
            
#         self.predictor = SAM2ImagePredictor.from_pretrained(
#             model_id=model_id_or_path,
#             device=device,
#             **kwargs
#         )

#     def __call__(
#         self, 
#         image: Image.Image,
#         **kwargs
#     ) -> Tuple[List[Image.Image], List[Optional[bool]]]:
#         try:
#             self.predictor.set_image(image)
#             prediction = self.predictor.predict(**kwargs)
#             return prediction
#         except Exception as e:
#             traceback.print_exc()
#             raise e

#     def __str__(self) -> str:
#         return f"Sam2Wrapper model_id={self.model_id}"