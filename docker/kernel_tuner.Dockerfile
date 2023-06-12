# CUDA version 10.2
FROM nvidia/cuda:10.2-devel-ubuntu18.04

WORKDIR /usr/bat/bat

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

# Copy content 
COPY . .

# Install dependencies for Kernel Tuner
RUN cd tuning_examples/kernel_tuner && \
    pip3 install -r requirements.txt

# Set the correct encoding for Python
ENV PYTHONIOENCODING=utf-8
