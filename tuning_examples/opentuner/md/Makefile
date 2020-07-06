
CC=g++
START_PATH=../../../src/programs

default: dependencies
	@echo "\033[93m[i]\033[0m Compiling... 😎"
	@nvcc -I $(START_PATH)/common/ -I $(START_PATH)/cuda-common/ -g -O2 -c -o main.o $(START_PATH)/cuda-common/main.cpp
	@nvcc -gencode=arch=compute_52,code=sm_52 -I $(START_PATH)/cuda-common -I $(START_PATH)/common -g -O2 -c $(START_PATH)/md/md.cu
	@nvcc -L $(START_PATH)/cuda-common -L $(START_PATH)/common -o md main.o md.o -lSHOCCommon
	@echo "\033[92m[✓]\033[0m Done compiling YAY! 🎉"

dependencies:
	@echo "\033[93m[i]\033[0m Compiling deps... 😎"
	@$(CC) -o $(START_PATH)/common/OptionParser.o -c $(START_PATH)/common/OptionParser.cpp
	@$(CC) -o $(START_PATH)/common/Option.o -c $(START_PATH)/common/Option.cpp
	@ar cr $(START_PATH)/common/libSHOCCommon.a $(START_PATH)/common/OptionParser.o $(START_PATH)/common/Option.o
	@ranlib $(START_PATH)/common/libSHOCCommon.a
	@echo "\033[92m[✓]\033[0m Done comiling deps! 🎉"

clean:
	@echo "\033[93m[i]\033[0m Cleanin' up...🗑️"
	@$(RM) *.o
	@$(RM) ./src/common/*.o
	@$(RM) ./src/common/libSHOCCommon.a
	@$(RM) ./triad
	@echo "\033[92m[✓]\033[0m Done cleaning up! 🎉🗑️"
