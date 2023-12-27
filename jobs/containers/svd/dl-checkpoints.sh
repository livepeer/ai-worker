#!/bin/bash

mkdir -p checkpoints
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir checkpoints