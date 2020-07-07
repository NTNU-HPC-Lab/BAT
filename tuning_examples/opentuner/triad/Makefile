
CC=g++
START_PATH=../../../src/programs

default: dependencies
	@echo "\033[93m[i]\033[0m Compiling... 🔨"
	@nvcc -I $(START_PATH)/common/ -I $(START_PATH)/cuda-common/ -g -O2 -c -o main.o $(START_PATH)/cuda-common/main.cpp
	@nvcc -gencode=arch=compute_52,code=sm_52 -I $(START_PATH)/cuda-common -I $(START_PATH)/common -g -O2 -c $(START_PATH)/triad/triad.cu
	@nvcc -gencode=arch=compute_52,code=sm_52 -I $(START_PATH)/cuda-common -I $(START_PATH)/common -g -O2 -c $(START_PATH)/triad/triad_kernel.cu
	@nvcc -L $(START_PATH)/cuda-common -L $(START_PATH)/common -o triad main.o triad.o -lSHOCCommon
	@echo "\033[92m[✓]\033[0m Done compiling program! 🎉"

dependencies:
	@echo "\033[93m[i]\033[0m Compiling dependencies... 🔨"
	@$(CC) -o $(START_PATH)/common/OptionParser.o -c $(START_PATH)/common/OptionParser.cpp
	@$(CC) -o $(START_PATH)/common/Option.o -c $(START_PATH)/common/Option.cpp
	@ar cr $(START_PATH)/common/libSHOCCommon.a $(START_PATH)/common/OptionParser.o $(START_PATH)/common/Option.o
	@ranlib $(START_PATH)/common/libSHOCCommon.a
	@echo "\033[92m[✓]\033[0m Done compiling dependencies! 🎉"

clean:
	@echo "\033[93m[i]\033[0m Cleaning up... 🗑️"
	@$(RM) *.o
	@$(RM) ./src/common/*.o
	@$(RM) ./src/common/libSHOCCommon.a
	@$(RM) ./triad
	@echo "\033[92m[✓]\033[0m Done cleaning up! 🎉"
