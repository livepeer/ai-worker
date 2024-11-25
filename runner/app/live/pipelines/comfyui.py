import os
import json
import torch
from PIL import Image
import asyncio
import numpy as np

from .interface import Pipeline
from comfystream.client import ComfyStreamClient

COMFY_UI_WORKSPACE_ENV = "COMFY_UI_WORKSPACE"
DEFAULT_WORKFLOW_JSON = '''
{
  "1": {
    "inputs": {
      "model": "sam2_hiera_tiny.safetensors",
      "segmentor": "realtime",
      "device": "cuda",
      "precision": "fp16"
    },
    "class_type": "DownloadAndLoadSAM2RealtimeModel",
    "_meta": {
      "title": "(Down)Load SAM2Model"
    }
  },
  "2": {
    "inputs": {
      "keep_model_loaded": true,
      "coordinates_positive": "[384,384]",
      "coordinates_negative": "",
      "individual_objects": false,
      "images": [
        "3",
        0
      ],
      "sam2_model": [
        "1",
        0
      ]
    },
    "class_type": "Sam2RealtimeSegmentation",
    "_meta": {
      "title": "Sam2RealtimeSegmentation"
    }
  },
  "3": {
    "inputs": {
      "image": "headroom.jpeg",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "4": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "2",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  }
}
'''


class ComfyUI(Pipeline):
  def __init__(self, **params):
    super().__init__(**params)

    comfy_ui_workspace = os.getenv(COMFY_UI_WORKSPACE_ENV)
    self.client = ComfyStreamClient(cwd=comfy_ui_workspace)

    params = {'prompt': json.loads(DEFAULT_WORKFLOW_JSON)}
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
    self.client.set_prompt(params['prompt'])
