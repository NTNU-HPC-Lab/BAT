# CUDA version 10.2
FROM nvidia/cuda:10.2-devel-ubuntu18.04

# TODO: fix this:
WORKDIR /usr/src/supercoolbenchmarksuite

RUN apt-get update && apt-get install -y \
    git \
    cmake

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

# Build all CLTune benchmarks
RUN cd tuning_examples/cltune/ \
    && cd sort && make

# Set the environment variable so other sources can use CLTune
ENV CLTUNE_PATH=/usr/local/CLTune
