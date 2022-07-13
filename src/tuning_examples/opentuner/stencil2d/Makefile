CC=g++
START_PATH=../../../src/programs

default: dependencies
	@echo "\033[93m[i]\033[0m Compiling... 🔨"
	@nvcc -I $(START_PATH)/common/ -I $(START_PATH)/cuda-common/ -g -O2 -c -o CUDAStencil.o $(START_PATH)/stencil2d/CUDAStencil.cpp
	@nvcc -I $(START_PATH)/common/ -I $(START_PATH)/cuda-common/ -g -O2 -c -o CommonCUDAStencilFactory.o $(START_PATH)/stencil2d/CommonCUDAStencilFactory.cpp
	@nvcc -I $(START_PATH)/common/ -I $(START_PATH)/cuda-common/ -g -O2 -c -o Stencil2Dmain.o $(START_PATH)/stencil2d/Stencil2Dmain.cpp
	@nvcc -I $(START_PATH)/common/ -I $(START_PATH)/cuda-common/ -g -O2 -c -o CUDAStencilFactory.o $(START_PATH)/stencil2d/CUDAStencilFactory.cpp
	@nvcc -I $(START_PATH)/common/ -I $(START_PATH)/cuda-common/ -g -O2 -c -o main.o $(START_PATH)/cuda-common/main.cpp
	@nvcc -gencode=arch=compute_52,code=sm_52 -I $(START_PATH)/cuda-common -I $(START_PATH)/common -g -O2 -c $(START_PATH)/stencil2d/CUDAStencilKernel.cu
	@nvcc -L $(START_PATH)/cuda-common -L $(START_PATH)/common -o Stencil2D CUDAStencil.o CommonCUDAStencilFactory.o Stencil2Dmain.o CUDAStencilFactory.o main.o CUDAStencilKernel.o -lSHOCCommon
	@echo "\033[92m[✓]\033[0m Done compiling program! 🎉"

dependencies:
	@echo "\033[93m[i]\033[0m Compiling dependencies... 🔨"
	@$(CC) -o $(START_PATH)/common/OptionParser.o -c $(START_PATH)/common/OptionParser.cpp
	@$(CC) -o $(START_PATH)/common/Option.o -c $(START_PATH)/common/Option.cpp
	@$(CC) -o $(START_PATH)/common/InvalidArgValue.o -c $(START_PATH)/common/InvalidArgValue.cpp
	@$(CC) -o $(START_PATH)/common/Matrix2DStatics.o -c $(START_PATH)/common/Matrix2DStatics.cpp
	@ar cr $(START_PATH)/common/libSHOCCommon.a $(START_PATH)/common/OptionParser.o $(START_PATH)/common/Option.o $(START_PATH)/common/InvalidArgValue.o $(START_PATH)/common/Matrix2DStatics.o
	@ranlib $(START_PATH)/common/libSHOCCommon.a
	@echo "\033[92m[✓]\033[0m Done compiling dependencies! 🎉"

clean:
	@echo "\033[93m[i]\033[0m Cleaning up... 🗑️"
	@$(RM) *.o
	@$(RM) $(START_PATH)/common/*.o
	@$(RM) $(START_PATH)/common/libSHOCCommon.a
	@$(RM) ./Stencil2D
	@echo "\033[92m[✓]\033[0m Done cleaning up! 🎉"
