#!/bin/bash

# Checks HF_TOKEN and huggingface-cli login status and throw warning if not authenticated.
check_hf_auth() {
    if [ -z "$HF_TOKEN" ] && [ "$(huggingface-cli whoami)" = "Not logged in" ]; then
        printf "WARN: Not logged in and HF_TOKEN not set. Log in with 'huggingface-cli login' or set HF_TOKEN to download token-gated models.\n"
        exit 1
    fi
}

# Displays help message.
function display_help() {
    echo "Description: This script is used to download models available on the Livepeer AI Subnet."
    echo "Usage: $0 [--beta]"
    echo "Options:"
    echo "  --beta  Download beta models."
    echo "  --restricted  Download models with a restrictive license."
    echo "  --help   Display this help message."
}

# Download recommended models during beta phase.
function download_beta_models() {
    printf "\nDownloading recommended beta phase models...\n"

    printf "\nDownloading unrestricted models...\n"

    # Download text-to-image and image-to-image models.
    huggingface-cli download SG161222/RealVisXL_V4.0_Lightning --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
    huggingface-cli download ByteDance/SDXL-Lightning --include "*unet.safetensors" --cache-dir models
    huggingface-cli download timbrooks/instruct-pix2pix --include "*.fp16.safetensors" "*.json" "*.txt" --cache-dir models

    # Download upscale models
    huggingface-cli download stabilityai/stable-diffusion-x4-upscaler --include "*.fp16.safetensors" --cache-dir models
    

    # Download FastSpeech 2 and HiFi-GAN models
    huggingface-cli download facebook/fastspeech2-en-ljspeech --include "*.bin" "*.json" --cache-dir models/fastspeech2
    huggingface-cli download facebook/hifigan --include "*.bin" "*.json" --cache-dir models/hifigan

    # Download audio-to-text models.
    huggingface-cli download openai/whisper-large-v3 --include "*.safetensors" "*.json" --cache-dir models

    # Download custom pipeline models.
    huggingface-cli download facebook/sam2-hiera-large --include "*.pt" "*.yaml" --cache-dir models

    printf "\nDownloading token-gated models...\n"

    # Download image-to-video models (token-gated).
    check_hf_auth
    huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt-1-1 --include "*.fp16.safetensors" "*.json" --cache-dir models ${TOKEN_FLAG:+"$TOKEN_FLAG"}
    

}

# Download all models.
function download_all_models() {
    download_beta_models

    printf "\nDownloading other available models...\n"

    # Download text-to-image and image-to-image models.
    printf "\nDownloading unrestricted models...\n"
    huggingface-cli download stabilityai/sd-turbo --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
    huggingface-cli download stabilityai/sdxl-turbo --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
    huggingface-cli download runwayml/stable-diffusion-v1-5 --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
    huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
    huggingface-cli download prompthero/openjourney-v4 --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
    huggingface-cli download SG161222/RealVisXL_V4.0 --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
    huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers --include "*.fp16*.safetensors" "*.model" "*.json" "*.txt" --cache-dir models ${TOKEN_FLAG:+"$TOKEN_FLAG"}
    huggingface-cli download SG161222/Realistic_Vision_V6.0_B1_noVAE --include "*.fp16.safetensors" "*.json" "*.txt" "*.bin" --exclude ".onnx" ".onnx_data" --cache-dir models
    huggingface-cli download black-forest-labs/FLUX.1-schnell --include "*.safetensors" "*.json" "*.txt" "*.model" --exclude ".onnx" ".onnx_data" --cache-dir models

    # Download image-to-video models.
    huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --include "*.fp16.safetensors" "*.json" --cache-dir models

    # Download image-to-text models.
    huggingface-cli download Salesforce/blip-image-captioning-large --include "*.safetensors" "*.json" --cache-dir models

    # Custom pipeline models.
    huggingface-cli download facebook/sam2-hiera-large --include "*.pt" "*.yaml" --cache-dir models

    # Download live-video-to-video models.
    huggingface-cli download KBlueLeaf/kohaku-v2.1 --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
    huggingface-cli download KwaiVGI/LivePortrait --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
}

# Download models with a restrictive license.
function download_restricted_models() {
    printf "\nDownloading restricted models...\n"

    # Download text-to-image and image-to-image models.
    huggingface-cli download black-forest-labs/FLUX.1-dev --include "*.safetensors" "*.json" "*.txt" "*.model" --exclude ".onnx" ".onnx_data" --cache-dir models ${TOKEN_FLAG:+"$TOKEN_FLAG"}
    # Download LLM models (Warning: large model size)
    huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "*.json" "*.bin" "*.safetensors" "*.txt" --cache-dir models

}

# Enable HF transfer acceleration.
# See: https://huggingface.co/docs/huggingface_hub/v0.22.1/package_reference/environment_variables#hfhubenablehftransfer.
export HF_HUB_ENABLE_HF_TRANSFER=1

# Use HF_TOKEN if set, otherwise use huggingface-cli's login.
[ -n "$HF_TOKEN" ] && TOKEN_FLAG="--token=${HF_TOKEN}" || TOKEN_FLAG=""

# Parse command-line arguments.
MODE="all"
for arg in "$@"
do
    case $arg in
        --beta)
            MODE="beta"
            shift
        ;;
        --restricted)
            MODE="restricted"
            shift
        ;;  
        --help)
            display_help
            exit 0
        ;;
        *)
            shift
        ;;
    esac
done

echo "Starting livepeer AI subnet model downloader..."
echo "Creating 'models' directory in the current working directory..."
mkdir -p models

# Ensure 'huggingface-cli' is installed.
echo "Checking if 'huggingface-cli' is installed..."
if ! command -v huggingface-cli > /dev/null 2>&1; then
    echo "WARN: The huggingface-cli is required to download models. Please install it using 'pip install huggingface_hub[cli,hf_transfer]'."
    exit 1
fi

if [ "$MODE" = "beta" ]; then
    download_beta_models
elif [ "$MODE" = "restricted" ]; then
    download_restricted_models
else
    download_all_models
fi

printf "\nAll models downloaded successfully!\n"
