# IDM-VTON Dockerfile for Vast.ai
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    ninja-build \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r api_requirements.txt
RUN pip3 install accelerate transformers diffusers einops bitsandbytes scipy opencv-python gradio fvcore cloudpickle omegaconf pycocotools basicsr av onnxruntime

# Install detectron2
RUN pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# Create necessary directories
RUN mkdir -p ckpt/densepose ckpt/humanparsing ckpt/openpose/ckpts ckpt/ip_adapter ckpt/image_encoder

# Download models (this will be done in the setup script)
# We'll download them during container startup to avoid huge image size

# Expose ports
EXPOSE 7860 8000

# Set up entry point
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]