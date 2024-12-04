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

    # Download audio-to-text models.
    huggingface-cli download openai/whisper-large-v3 --include "*.safetensors" "*.json" --cache-dir models
    huggingface-cli download distil-whisper/distil-large-v3 --include "*.safetensors" "*.json" --cache-dir models
    huggingface-cli download openai/whisper-medium --include "*.safetensors" "*.json" --cache-dir models

    # Download custom pipeline models.
    huggingface-cli download facebook/sam2-hiera-large --include "*.pt" "*.yaml" --cache-dir models
    huggingface-cli download parler-tts/parler-tts-large-v1 --include "*.safetensors" "*.json" "*.model" --cache-dir models

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
    huggingface-cli download stabilityai/stable-diffusion-3.5-medium --include "transformer/*.safetensors" "*model.fp16*" "*.model" "*.json" "*.txt" --cache-dir models ${TOKEN_FLAG:+"$TOKEN_FLAG"}
    huggingface-cli download stabilityai/stable-diffusion-3.5-large --include "transformer/*.safetensors" "*model.fp16*" "*.model" "*.json" "*.txt" --cache-dir models ${TOKEN_FLAG:+"$TOKEN_FLAG"}
    huggingface-cli download SG161222/Realistic_Vision_V6.0_B1_noVAE --include "*.fp16.safetensors" "*.json" "*.txt" "*.bin" --exclude ".onnx" ".onnx_data" --cache-dir models
    huggingface-cli download black-forest-labs/FLUX.1-schnell --include "*.safetensors" "*.json" "*.txt" "*.model" --exclude ".onnx" ".onnx_data" --cache-dir models

    # Download image-to-video models.
    huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --include "*.fp16.safetensors" "*.json" --cache-dir models

    # Download image-to-text models.
    huggingface-cli download Salesforce/blip-image-captioning-large --include "*.safetensors" "*.json" --cache-dir models

    # Custom pipeline models.
    huggingface-cli download facebook/sam2-hiera-large --include "*.pt" "*.yaml" --cache-dir models

    download_live_models
}

# Download models only for the live-video-to-video pipeline.
function download_live_models() {
    huggingface-cli download KBlueLeaf/kohaku-v2.1 --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
    huggingface-cli download stabilityai/sd-turbo --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
    huggingface-cli download warmshao/FasterLivePortrait --local-dir models/FasterLivePortrait--checkpoints
    huggingface-cli download yuvraj108c/Depth-Anything-Onnx --include depth_anything_vitl14.onnx --local-dir models/ComfyUI--models/Depth-Anything-Onnx
    download_sam2_checkpoints
}

function download_sam2_checkpoints() {
    huggingface-cli download facebook/sam2-hiera-tiny --local-dir models/sam2--checkpoints/facebook--sam2-hiera-tiny
    huggingface-cli download facebook/sam2-hiera-small --local-dir models/sam2--checkpoints/facebook--sam2-hiera-small
    huggingface-cli download facebook/sam2-hiera-large --local-dir models/sam2--checkpoints/facebook--sam2-hiera-large
}

function build_tensorrt_models() {
    download_live_models

    printf "\nBuilding TensorRT models...\n"

    # StreamDiffusion (compile a matrix of models and timesteps)
    MODELS="stabilityai/sd-turbo KBlueLeaf/kohaku-v2.1"
    TIMESTEPS="3 4" # This is basically the supported sizes for the t_index_list
    docker run --rm -it -v ./models:/models --gpus all \
        livepeer/ai-runner:live-app-streamdiffusion \
        bash -c "for model in $MODELS; do
                    for timestep in $TIMESTEPS; do
                        echo \"Building TensorRT engines for model=\$model timestep=\$timestep...\" && \
                        python app/live/StreamDiffusionWrapper/build_tensorrt.py --model-id \$model --timesteps \$timestep
                    done
                done"

    # FasterLivePortrait
    docker run --rm -it -v ./models:/models --gpus all \
        livepeer/ai-runner:live-app-liveportrait \
        bash -c "cd /app/app/live/FasterLivePortrait && \
                    if [ ! -f '/models/FasterLivePortrait--checkpoints/liveportrait_onnx/stitching_lip.trt' ]; then
                        echo 'Building TensorRT engines for LivePortrait models (regular)...'
                        sh scripts/all_onnx2trt.sh
                    else
                        echo 'Regular LivePortrait TensorRT engines already exist, skipping build'
                    fi && \
                    if [ ! -f '/models/FasterLivePortrait--checkpoints/liveportrait_animal_onnx/stitching_lip.trt' ]; then
                        echo 'Building TensorRT engines for LivePortrait models (animal)...'
                        sh scripts/all_onnx2trt_animal.sh
                    else
                        echo 'Animal LivePortrait TensorRT engines already exist, skipping build'
                    fi"

    # ComfyUI (only DepthAnything for now)
    docker run --rm -it -v ./models:/models --gpus all \
        livepeer/ai-runner:live-app-comfyui \
        bash -c "cd /comfyui/models/Depth-Anything-Onnx && \
                    python /comfyui/custom_nodes/ComfyUI-Depth-Anything-Tensorrt/export_trt.py && \
                    mkdir -p /comfyui/models/tensorrt/depth-anything && \
                    mv *.engine /comfyui/models/tensorrt/depth-anything"
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
        --live)
            MODE="live"
            shift
        ;;
        --tensorrt)
            MODE="tensorrt"
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
mkdir -p models/StreamDiffusion--engines models/FasterLivePortrait--checkpoints models/ComfyUI--models models/sam2--checkpoints

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
elif [ "$MODE" = "live" ]; then
    download_live_models
elif [ "$MODE" = "tensorrt" ]; then
    build_tensorrt_models
else
    download_all_models
fi

printf "\nAll models downloaded successfully!\n"
