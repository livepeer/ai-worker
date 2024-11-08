from comfystream.client import ComfyStreamClient

import os
import json
import torch
from PIL import Image


class ComfyPipeline(Pipeline):
  def __init__(self, **params):
    # configure as is needed
    comfy_ui_workspace = os.getenv("COMFY_UI_WORKSPACE")
    default_workflow = os.genenv("COMFY_UI_DEFAULT_WORKFLOW")

    self.client = ComfyStreamClient(comfy_ui_workspace)
    with open(default_workflow, "r") as f:
      prompt = json.load(f)
    self.client.set_prompt(prompt)

    # Comfy will cache nodes that only need to be run once (i.e. a node that loads model weights)
    # We can run the prompt once before actual inputs come in to "warmup"
    input = torch.randn(1, 512, 512, 3)
    self.client.queue_prompt(input)

    self.update_params(**params)

  async def process_frame(self, image: Image.Image) -> Image.Image:
    # preprocess image into a torch.Tensor
    output = self.client.queue_prompt(input)
    # postprocess output into a Image.Image
    return output

  def update_params(self, **params):
    # Convert params into a Prompt type which describes the workflow
    self.client.set_prompt(params["config"])
  