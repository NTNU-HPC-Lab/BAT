# CUDA version 10.2
FROM nvidia/cuda:10.2-devel-ubuntu18.04

WORKDIR /usr/src/BFS

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    openmpi-bin \
    openssh-client \
    libopenmpi-dev

RUN pip3 install \
    https://github.com/ingunnsund/opentuner/archive/master.zip \
    numba

# Copy content 
COPY . .

RUN make clean
RUN make dependencies

#nvidia-docker build -t bfs .
#nvidia-docker run --rm -ti bfs

#CMD python3 BFS_tuner.py --no-dups --stop-after=30 