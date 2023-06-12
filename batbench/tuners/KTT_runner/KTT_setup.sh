#!/bin/bash
PYTHON_HEADERS=/usr/include/python3.10 PYTHON_LIB=/lib/x86_64-linux-gnu/libpython3.10.so ./premake5 --platform=nvidia --python --tuning-loader gmake && cd Build && make config=release_x86_64

