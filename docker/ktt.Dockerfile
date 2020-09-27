# CUDA version 10.2
FROM nvidia/cuda:10.2-devel-ubuntu18.04

WORKDIR /usr/src/bat

RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3

# Download premake5 (dependency of KTT)
RUN wget https://github.com/premake/premake-core/releases/download/v5.0.0-alpha15/premake-5.0.0-alpha15-linux.tar.gz \
    && tar -xzf premake-5.0.0-alpha15-linux.tar.gz \
    && rm premake-5.0.0-alpha15-linux.tar.gz \
    && mv premake5 /usr/bin

# Set CUDA path required by KTT
ENV CUDA_PATH=/usr/local/cuda-10.2/

# Add temporary linking path for building KTT
ARG TEMP_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${CUDA_PATH}lib64/stubs:${LD_LIBRARY_PATH}

# Create symbolic link for CUDA libraries to be used during building
# This is due to libraries being different in build and run phase of Docker (See issue: https://github.com/NVIDIA/nvidia-docker/issues/775)
# This is for linking to work on Docker build for CUDA libs required by KTT
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Download and build KTT
RUN cd /usr/local \
    && git clone https://github.com/Fillo7/KTT \
    && cd KTT \
    && premake5 gmake \
    && cd build \
    && make config=release_x86_64

# Copy content 
COPY . .

# Build all KTT benchmarks
RUN cd tuning_examples/ktt/ \
    && cd sort && make

# Remove the symbolic link for the CUDA libraries as it is not needed anymore
RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1

# Reset the LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${TEMP_LD_LIBRARY_PATH}

# Set the environment variable so other sources can use KTT
ENV KTT_PATH=/usr/local/KTT

# Set the correct encoding for Python
ENV PYTHONIOENCODING=utf-8
