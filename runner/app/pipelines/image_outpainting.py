from diffusers import AutoPipelineForInpainting
import torch
from PIL import Image
import numpy as np

class ImageOutpaintingPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            # Use AutoPipelineForInpainting to load ProPainter
            self.pipe = AutoPipelineForInpainting.from_pretrained("ruffy369/propainter", torch_dtype=torch.float16).to(self.device)
            print("ProPainter model loaded successfully.")
        except Exception as e:
            print(f"Error loading ProPainter model: {e}")
            self.pipe = None

    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ):
        if self.pipe is None:
            print("ProPainter model is not loaded. Cannot perform outpainting.")
            return None

        # Prepare the image for outpainting
        width, height = image.size
        target_size = min(max(width, height) * 2, 1024)  # Double the size, but cap at 1024
        new_image = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        new_image.paste(image, ((target_size - width) // 2, (target_size - height) // 2))
        
        # Create a mask for outpainting
        mask = Image.new('L', (target_size, target_size), 255)
        mask.paste(0, ((target_size - width) // 2, (target_size - height) // 2, 
                       (target_size + width) // 2, (target_size + height) // 2))

        try:
            # Generate the outpainted image
            output = self.pipe(
                prompt=prompt,
                image=new_image,
                mask_image=mask,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

            return output
        except Exception as e:
            print(f"Error during outpainting: {e}")
            return None
