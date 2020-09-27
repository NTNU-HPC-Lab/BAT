# CUDA version 10.2
FROM nvidia/cuda:10.2-devel-ubuntu18.04

WORKDIR /usr/src/bat

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    openmpi-bin \
    openssh-client \
    libopenmpi-dev

# Copy content 
COPY . .

RUN cd tuning_examples/opentuner && \
    pip3 install -r requirements.txt

# Set the correct encoding for Python
ENV PYTHONIOENCODING=utf-8
