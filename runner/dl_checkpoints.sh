#!/bin/bash
echo "Starting livepeer AI subnet model downloader..."

# Enable HF transfer acceleration.
# See: https://huggingface.co/docs/huggingface_hub/v0.22.1/package_reference/environment_variables#hfhubenablehftransfer.
export HF_HUB_ENABLE_HF_TRANSFER=1

# Use HF_TOKEN if set, otherwise use huggingface-cli's login.
[ -n "$HF_TOKEN" ] && TOKEN_FLAG="--token=${HF_TOKEN}" || TOKEN_FLAG=""

echo "Creating models directory in the current working directory..."
mkdir -p models

# Ensure 'huggingface-cli' is installed.
echo "Checking if 'huggingface-cli' is installed..."
if ! command -v huggingface-cli > /dev/null 2>&1; then
    echo "WARN: The huggingface-cli is required to download models. Please install it using 'pip install huggingface_hub[cli,hf_transfer]'."
    exit
fi

# Download text-to-image and image-to-image models.
printf "\nDownloading unrestricted models...\n"
huggingface-cli download stabilityai/sd-turbo --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
huggingface-cli download stabilityai/sdxl-turbo --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
huggingface-cli download runwayml/stable-diffusion-v1-5 --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
huggingface-cli download prompthero/openjourney-v4 --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
huggingface-cli download ByteDance/SDXL-Lightning --include "*unet.safetensors" --exclude "*lora.safetensors*" --cache-dir models

# Download image-to-video models.
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --include "*.fp16.safetensors" "*.json" --cache-dir models

# Ensure user is logged in or has HF_TOKEN set to download token-gated models.
printf "\nDownloading token-gated models...\n"
if [ -z "$HF_TOKEN" ] && [ "$(huggingface-cli whoami)" = "Not logged in" ]; then
    printf "WARN: Not logged in and HF_TOKEN not set. Log in with 'huggingface-cli login' or set HF_TOKEN to download token-gated models.\n"
    exit
fi

# Download image-to-video models (token-gated).
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt-1-1 --include "*.fp16.safetensors" "*.json" --cache-dir models ${TOKEN_FLAG}

printf "\nAll models downloaded successfully!\n"
