CC=g++
START_PATH=../../../src/programs

default: dependencies
	@echo "\033[93m[i]\033[0m Compiling... 😎"
	@nvcc -I $(START_PATH)/common/ -I $(START_PATH)/cuda-common/ -g -O2 -c -o Graph.o $(START_PATH)/common/Graph.cpp
	@nvcc -I $(START_PATH)/common/ -I $(START_PATH)/cuda-common/ -g -O2 -c -o main.o $(START_PATH)/cuda-common/main.cpp
	@nvcc -gencode=arch=compute_52,code=sm_52 -I $(START_PATH)/cuda-common -I $(START_PATH)/common -g -O2 -c $(START_PATH)/bfs/BFS.cu
	@nvcc -gencode=arch=compute_52,code=sm_52 -I $(START_PATH)/cuda-common -I $(START_PATH)/common -g -O2 -c $(START_PATH)/bfs/bfs_kernel.cu
	@nvcc -L $(START_PATH)/cuda-common -L $(START_PATH)/common -o BFS Graph.o main.o BFS.o bfs_kernel.o -lSHOCCommon
	@echo "\033[92m[✓]\033[0m Done compiling YAY! 🎉"

parallel: dependencies
	@echo "\033[93m[i]\033[0m Compiling... 😎"
	@mpicxx -I $(START_PATH)/common/ -I $(START_PATH)/cuda-common/ -I /usr/local/cuda/include -DPARALLEL  -I $(START_PATH)/mpi-common/ -g -O2 -c -o Graph.o $(START_PATH)/common/Graph.cpp
	@mpicxx -I $(START_PATH)/common/ -I $(START_PATH)/cuda-common/ -I /usr/local/cuda/include -DPARALLEL  -I $(START_PATH)/mpi-common/ -g -O2 -c -o main.o $(START_PATH)/cuda-common/main.cpp
	@nvcc -gencode=arch=compute_52,code=sm_52 -I $(START_PATH)/cuda-common -I $(START_PATH)/common -g -O2 -c $(START_PATH)/bfs/BFS.cu
	@nvcc -gencode=arch=compute_52,code=sm_52 -I $(START_PATH)/cuda-common -I $(START_PATH)/common -g -O2 -c $(START_PATH)/bfs/bfs_kernel.cu
	@mpicxx -L $(START_PATH)/cuda-common -L $(START_PATH)/common -o BFS Graph.o main.o BFS.o bfs_kernel.o -lSHOCCommon "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lrt -lrt
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
	@$(RM) $(START_PATH)/common/*.o
	@$(RM) $(START_PATH)/common/libSHOCCommon.a
	@$(RM) ./BFS
	@echo "\033[92m[✓]\033[0m Done cleaning up! 🎉🗑️"
