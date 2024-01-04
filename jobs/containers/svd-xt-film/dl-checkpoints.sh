#!/bin/bash

mkdir -p cache
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --include "*.fp16.safetensors" "*.json" --cache-dir cache 
wget -O cache/film_net_fp16.pt https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.2/film_net_fp16.pt