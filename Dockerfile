ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# fetch the key refer to https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/9
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 32
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="(dirname(which conda))/../"

RUN apt-get update && apt-get install -y \
    libglu1-mesa-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    htop \
    unzip    \
    graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MIM
RUN pip install openmim

COPY requirements.txt /workspace/requirements.txt

# Install pip packages
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /workspace/requirements.txt

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the serve program are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/workspace:${PATH}"
# Configure python path for custom modules
ENV PYTHONPATH "${PYTHONPATH}:/workspace"

# Configure default shell
ENV SHELL=/bin/bash

RUN mkdir /opt/mmopenlab

# Install MMPretrain
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git && mv mmdetection /opt/mmopenlab
RUN git clone https://github.com/open-mmlab/mmpretrain.git  && mv mmpretrain /opt/mmopenlab

RUN cd /opt/mmopenlab/mmdetection && mim install --no-cache-dir -e .
RUN cd /opt/mmopenlab/mmpretrain && mim install --no-cache-dir -e .

WORKDIR /workspace