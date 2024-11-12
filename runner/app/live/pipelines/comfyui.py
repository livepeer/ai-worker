from comfystream.client import ComfyStreamClient

import os
import json
import torch
from PIL import Image
import asyncio
import numpy as np

from .interface import Pipeline


class ComfyUI(Pipeline):
  def __init__(self, **params):
    super().__init__(**params)
    # configure as is needed
    comfy_ui_workspace = os.getenv("COMFY_UI_WORKSPACE")
    default_workflow = os.getenv("COMFY_UI_DEFAULT_WORKFLOW")

    self.client = ComfyStreamClient(cwd=comfy_ui_workspace)
    with open(default_workflow, "r") as f:
      prompt = json.load(f)
    self.client.set_prompt(prompt)

    # Comfy will cache nodes that only need to be run once (i.e. a node that loads model weights)
    # We can run the prompt once before actual inputs come in to "warmup"
    # input = torch.randn(1, 512, 512, 3)
    # self.client.queue_prompt(input)

    # self.update_params(**params)

  def process_frame(self, image: Image.Image) -> Image.Image:
    # Normalize by dividing by 255 to ensure the tensor values are between 0 and 1
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    # Convert from numpy to torch.Tensor
    # Initially, the torch.Tensor will have shape HWC but we want BHWC
    # unsqueeze(0) will add a batch dimension at the beginning of 1 which means we just have 1 image
    image_tensor = torch.tensor(image_np).unsqueeze(0)

    # Process using ComfyUI pipeline
    loop = asyncio.get_event_loop()
    result_tensor = loop.run_until_complete(self.client.queue_prompt(image_tensor))

    # Convert back from Tensor to PIL.Image
    result_tensor = result_tensor.squeeze(0)
    result_image_np = (result_tensor * 255).byte()
    result_image = Image.fromarray(result_image_np.cpu().numpy())
    return result_image

  def update_params(self, **params):
    # Convert params into a Prompt type which describes the workflow
    # self.client.set_prompt(params["config"])
    return
