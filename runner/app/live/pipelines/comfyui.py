from comfystream.client import ComfyStreamClient

import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import asyncio

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
    # transform = transforms.ToTensor()
    transform = transforms.Compose([
      transforms.Resize((512, 512)),  # Resize the image to 512x512
      transforms.ToTensor(),          # Convert the image to a tensor
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    # Apply the transformation to convert to torch.Tensor
    if image.mode != 'RGB':
      image = image.convert('RGB')
    tensor = transform(image)
    if tensor.dim() == 3:  # If there's no batch dimension
      tensor = tensor.unsqueeze(0)
    conv_layer = nn.Conv2d(3, 512, kernel_size=1)
    extended_tensor = conv_layer(tensor) 
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(self.client.queue_prompt(extended_tensor))

    # to_pil = transforms.ToPILImage()
    # result_image = to_pil(result)
    # preprocess image into a torch.Tensor
    # output = self.client.queue_prompt(input)
    # postprocess output into a Image.Image
    # return output
    return image.convert("RGB")

  def update_params(self, **params):
    # Convert params into a Prompt type which describes the workflow
    # self.client.set_prompt(params["config"])
    return
