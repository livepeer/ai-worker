from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from PIL import Image
import io
from ..pipelines.image_outpainting import ImageOutpaintingPipeline
from .util import ImageOutpaintingResponse


router = APIRouter()
pipeline = ImageOutpaintingPipeline()



def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize image while maintaining aspect ratio if it exceeds max_size"""
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    return image

@router.post("/out-paint", response_model=ImageOutpaintingResponse)
async def out_paint(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(None),
    num_inference_steps: int = Form(50, ge=1, le=1000),
    guidance_scale: float = Form(7.5, ge=0, le=20),
):
    if len(prompt) > 1000:
        raise HTTPException(status_code=400, detail="Prompt is too long")
    if negative_prompt and len(negative_prompt) > 1000:
        raise HTTPException(status_code=400, detail="Negative prompt is too long")
    
    try:
        image_content = await image.read()
        input_image = resize_image(Image.open(io.BytesIO(image_content)).convert("RGB"))
        
        output_image = pipeline(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        # Convert the output image to bytes for response
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return ImageOutpaintingResponse(
            image=img_byte_arr,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during outpainting: {str(e)}")
