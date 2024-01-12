#!/bin/bash

mkdir -p models

# text-to-image, image-to-image
huggingface-cli download stabilityai/sd-turbo --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
# image-to-video
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --include "*.fp16.safetensors" "*.json" --cache-dir models