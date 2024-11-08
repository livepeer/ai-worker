from comfystream.client import ComfyStreamClient

class ComfyPipeline(Pipeline):
  def __init__(self, **params):
    # configure as is needed
    self.client = ComfyStreamClient(...)
    self.update_params(**params)

  async def process_frame(self, image: Image.Image) -> Image.Image:
    # preprocess image into a torch.Tensor
    output = self.client.queue_prompt(input)
    # postprocess output into a Image.Image
    return output

  def update_params(self, **params):
    # Convert params into a Prompt type which describes the workflow
    self.client.set_prompt(prompt)

  ...