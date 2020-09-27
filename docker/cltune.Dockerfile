# CUDA version 10.2
FROM nvidia/cuda:10.2-devel-ubuntu18.04

WORKDIR /usr/src/bat

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    python3

# Download and build CLTune
RUN cd /usr/local \
    && git clone https://github.com/CNugteren/CLTune \
    && cd CLTune \
    && mkdir build \
    && cd build \
    && cmake -DUSE_OPENCL=OFF .. \
    && make install

# Copy content 
COPY . .

# Set the environment variable so other sources can use CLTune
ENV CLTUNE_PATH=/usr/local/CLTune

# Set the correct encoding for Python
ENV PYTHONIOENCODING=utf-8
