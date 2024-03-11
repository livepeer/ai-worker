#!/bin/bash

mkdir -p models

# Use HF_TOKEN if set, otherwise use huggingface-cli's login.
[ -n "$HF_TOKEN" ] && TOKEN_FLAG="--token=${HF_TOKEN}" || TOKEN_FLAG=""
echo $TOKEN_FLAG

# text-to-image, image-to-image
echo "Downloading unrestricted models..."
huggingface-cli download stabilityai/sd-turbo --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
huggingface-cli download stabilityai/sdxl-turbo --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
huggingface-cli download runwayml/stable-diffusion-v1-5 --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
huggingface-cli download prompthero/openjourney-v4 --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
huggingface-cli download ByteDance/SDXL-Lightning --include "*unet.safetensors" --exclude "*lora.safetensors*" --cache-dir models

# image-to-video
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --include "*.fp16.safetensors" "*.json" --cache-dir models

# image-to-video (token-gated)
echo -e "\nDownloading token-gated models..." 
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt-1-1 --include "*.fp16.safetensors" "*.json" --cache-dir models ${TOKEN_FLAG}

echo "Models downloaded successfully!"
