import os
import json
import torch
from PIL import Image
import asyncio
import numpy as np

from .interface import Pipeline
from comfystream.client import ComfyStreamClient

import logging

COMFY_UI_WORKSPACE_ENV = "COMFY_UI_WORKSPACE"
DEFAULT_WORKFLOW_JSON = '''
{
    "1": {
        "inputs": {
            "images": ["2", 0]
        },
        "class_type": "SaveTensor",
        "_meta": {
            "title": "SaveTensor"
        }
    },
    "2": {
        "inputs": {
            "engine": "depth_anything_vitl14-fp16.engine",
            "images": ["3", 0]
        },
        "class_type": "DepthAnythingTensorrt",
        "_meta": {
            "title": "Depth Anything Tensorrt"
        }
    },
    "3": {
        "inputs": {},
        "class_type": "LoadTensor",
        "_meta": {
            "title": "LoadTensor"
        }
    }
}
'''


class ComfyUI(Pipeline):
  def __init__(self, **params):
    super().__init__(**params)

    comfy_ui_workspace = os.getenv(COMFY_UI_WORKSPACE_ENV)
    self.client = ComfyStreamClient(cwd=comfy_ui_workspace)

    params = {
        'prompt': params['prompt'] if params.get('prompt') not in (None, "") else json.loads(DEFAULT_WORKFLOW_JSON)
    }

    self.update_params(**params)

    # Comfy will cache nodes that only need to be run once (i.e. a node that loads model weights)
    # We can run the prompt once before actual inputs come in to "warmup"
    warmup_input = torch.randn(1, 512, 512, 3)
    asyncio.get_event_loop().run_until_complete(self.client.queue_prompt(warmup_input))

  def process_frame(self, image: Image.Image) -> Image.Image:
    # Normalize by dividing by 255 to ensure the tensor values are between 0 and 1
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    # Convert from numpy to torch.Tensor
    # Initially, the torch.Tensor will have shape HWC but we want BHWC
    # unsqueeze(0) will add a batch dimension at the beginning of 1 which means we just have 1 image
    image_tensor = torch.tensor(image_np).unsqueeze(0)

    # Process using ComfyUI pipeline
    result_tensor = asyncio.get_event_loop().run_until_complete(self.client.queue_prompt(image_tensor))

    # Convert back from Tensor to PIL.Image
    result_tensor = result_tensor.squeeze(0)
    result_image_np = (result_tensor * 255).byte()
    result_image = Image.fromarray(result_image_np.cpu().numpy())
    return result_image

  def update_params(self, **params):
    # params['prompt'] is the JSON string with the ComfyUI workflow
    logging.info(f"ComfyUI Pipeline Prompt: {params['prompt']}")
    self.client.set_prompt(params['prompt'])
