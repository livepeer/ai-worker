# Based on https://github.com/huggingface/api-inference-community/blob/main/docker_images/diffusers/Dockerfile
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites
RUN apt-get update && \
    apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git runit libzmq3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl https://pyenv.run | bash

# Set environment variables for pyenv
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install your desired Python version
ARG PYTHON_VERSION=3.11
RUN pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash

# Upgrade pip and install your desired packages
ARG PIP_VERSION=23.3.2
RUN pip install --no-cache-dir --upgrade pip==${PIP_VERSION} setuptools==69.5.1 wheel==0.43.0 

# Set environment variables
ENV MAX_WORKERS=1
ENV HUGGINGFACE_HUB_CACHE=/models
ENV DIFFUSERS_CACHE=/models
ENV MODEL_DIR=/models

# The following steps have been moved to the app Dockerfile:
# - Copying application files
# - Setting up and compiling Go application
# - Setting up runit service directories
# - Creating log directories
# - Setting the init system to runit

# We're keeping the WORKDIR /app here as it's a good default
WORKDIR /app