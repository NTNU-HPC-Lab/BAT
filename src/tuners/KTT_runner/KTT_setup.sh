./premake5 --platform=nvidia --python --tuning-loader gmake
cd Build && PYTHON_HEADERS=/usr/include/python3.8 PYTHON_LIB=/lib/x86_64-linux-gnu/libpython3.8.so make config=release_x86_64

