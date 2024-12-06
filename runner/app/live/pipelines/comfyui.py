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
  "27": {
    "inputs": {
      "image": "chillguy.png",
      "upload": "image"
    },
    "class_type": "LoadImage"
  },
  "52": {
    "inputs": {
      "images": [
        "73",
        0
      ]
    },
    "class_type": "PreviewImage"
  },
  "57": {
    "inputs": {
      "warmup": 10,
      "do_add_noise": true,
      "use_denoising_batch": true
    },
    "class_type": "StreamDiffusionAccelerationConfig"
  },
  "61": {
    "inputs": {
      "prompt": "Realistic transformation of a human face into a futuristic silver robot. The face should retain recognizable human features but with metallic silver skin, glowing blue eyes, and subtle mechanical details like circuits and panels integrated seamlessly. The overall look should be sleek and modern, with a shiny chrome finish and a slight reflection of light. The background is neutral and futuristic, softly lit to enhance the metallic details. Artistic and photorealistic style, highly detailed, ultra-sharp focus, 8k resolution.",
      "negative_prompt": "Blurry details, low quality, cartoonish style, unrealistic features, disfigured face, excessive noise, messy background, overly dark shadows, unnatural proportions, dull or matte finish, extra limbs, distorted facial structure, overly dramatic lighting, grainy texture.",
      "num_inference_steps": 50,
      "guidance_scale": 1.2,
      "delta": 1,
      "stream_model": [
        "64",
        0
      ],
      "image": [
        "27",
        0
      ]
    },
    "class_type": "StreamDiffusionAccelerationSampler"
  },
  "64": {
    "inputs": {
      "t_index_list": "39,35,30",
      "mode": "img2img",
      "width": 512,
      "height": 512,
      "acceleration": "tensorrt",
      "frame_buffer_size": 1,
      "use_tiny_vae": true,
      "cfg_type": "self",
      "use_lcm_lora": true,
      "model": [
        "65",
        0
      ],
      "opt_acceleration_config": [
        "57",
        0
      ]
    },
    "class_type": "StreamDiffusionConfig"
  },
  "65": {
    "inputs": {
      "model_id_or_path": "KBlueLeaf/kohaku-v2.1"
    },
    "class_type": "StreamDiffusionModelLoader"
  },
  "73": {
    "inputs": {
      "x": 0,
      "y": 0,
      "resize_source": false,
      "destination": [
        "27",
        0
      ],
      "source": [
        "61",
        0
      ],
      "mask": [
        "75",
        1
      ]
    },
    "class_type": "ImageCompositeMasked"
  },
  "74": {
    "inputs": {
      "model": "sam2_hiera_tiny.pt",
      "segmentor": "realtime",
      "device": "cuda",
      "precision": "fp16"
    },
    "class_type": "DownloadAndLoadSAM2RealtimeModel"
  },
  "75": {
    "inputs": {
      "reset_tracking": false,
      "coordinates_positive": "[[384,384]]",
      "coordinates_negative": "[[50,50]]",
      "images": [
        "27",
        0
      ],
      "sam2_model": [
        "74",
        0
      ]
    },
    "class_type": "Sam2RealtimeSegmentation"
  },
  "76": {
    "inputs": {
      "images": [
        "77",
        0
      ]
    },
    "class_type": "PreviewImage"
  },
  "77": {
    "inputs": {
      "mask": [
        "75",
        1
      ]
    },
    "class_type": "MaskToImage"
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
