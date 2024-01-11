#!/bin/bash

mkdir -p cache
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --include "*.fp16.safetensors" "*.json" "*.txt" --cache-dir cache
huggingface-cli download stabilityai/stable-diffusion-xl-refiner-1.0 --include "*.fp16.safetensors" "*.json" "*.txt" --cache-dir cache