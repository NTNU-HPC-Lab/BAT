# CUDA version 10.2
FROM nvidia/cuda:10.2-devel-ubuntu18.04

WORKDIR /usr/bat/bat

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    openmpi-bin \
    openssh-client \
    libopenmpi-dev

# Copy content 
COPY . .

# Install dependencies for OpenTuner
RUN cd tuning_examples/opentuner && \
    pip3 install -r requirements.txt

# Due to a bug in OpenMPI (https://github.com/open-mpi/ompi/issues/4948)
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none

# Set the correct encoding for Python
ENV PYTHONIOENCODING=utf-8
