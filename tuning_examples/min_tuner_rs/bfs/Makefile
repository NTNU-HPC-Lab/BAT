CC=g++
START_PATH=../../../src/programs
COMPUTE_CAPABILITY=$(shell nvidia-smi --query-gpu=compute_cap --format=csv | tail -1 | sed 's/\.//g')
default: $(START_PATH)/common/libSHOCCommon.a Graph.o main.o BFS.o
	echo "\033[93m[i]\033[0m Compiling... 🔨 (default)"
	echo "$(PARAMETERS)"
	echo "$(BIN_NAME)"
	echo "$(COMPUTE_CAPABILITY)"
	nvcc -gencode=arch=compute_$(COMPUTE_CAPABILITY),code=sm_$(COMPUTE_CAPABILITY) -I $(START_PATH)/cuda-common -I $(START_PATH)/common -o bfs_kernel.o -O2 -c $(START_PATH)/bfs/bfs_kernel.cu $(PARAMETERS)
	nvcc -L $(START_PATH)/cuda-common -L $(START_PATH)/common -o ${BIN_NAME} Graph.o main.o BFS.o bfs_kernel.o -lSHOCCommon
	echo "\033[92m[✓]\033[0m Done compiling program! 🎉"

parallel: dependencies
	echo "\033[93m[i]\033[0m Compiling parallel... 🔨"
	mpicxx -I $(START_PATH)/common/ -I $(START_PATH)/cuda-common/ -I /usr/local/cuda/include -DPARALLEL  -I $(START_PATH)/mpi-common/ -g -O2 -c -o Graph.o $(START_PATH)/common/Graph.cpp 
	mpicxx -I $(START_PATH)/common/ -I $(START_PATH)/cuda-common/ -I /usr/local/cuda/include -DPARALLEL  -I $(START_PATH)/mpi-common/ -g -O2 -c -o main.o $(START_PATH)/cuda-common/main.cpp
	nvcc -gencode=arch=compute_52,code=sm_52 -I $(START_PATH)/cuda-common -I $(START_PATH)/common -g -O2 -c $(START_PATH)/bfs/BFS.cu $(PARAMETERS)
	nvcc -gencode=arch=compute_52,code=sm_52 -I $(START_PATH)/cuda-common -I $(START_PATH)/common -g -O2 -c $(START_PATH)/bfs/bfs_kernel.cu $(PARAMETERS)
	mpicxx -L $(START_PATH)/cuda-common -L $(START_PATH)/common -o ${BIN_NAME} Graph.o main.o BFS.o -lSHOCCommon "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lrt -lrt
	echo "\033[92m[✓]\033[0m Done compiling parallel program! 🎉"

$(START_PATH)/common/OptionParser.o:
	$(CC) -o $(START_PATH)/common/OptionParser.o -c $(START_PATH)/common/OptionParser.cpp

$(START_PATH)/common/Option.o:
	$(CC) -o $(START_PATH)/common/Option.o -c $(START_PATH)/common/Option.cpp

Graph.o:
	nvcc -I $(START_PATH)/common/ -I $(START_PATH)/cuda-common/  -O2 -c -o Graph.o $(START_PATH)/common/Graph.cpp

main.o:
	nvcc -I $(START_PATH)/common/ -I $(START_PATH)/cuda-common/  -O2 -c -o main.o $(START_PATH)/cuda-common/main.cpp

BFS.o:
	nvcc -gencode=arch=compute_$(COMPUTE_CAPABILITY),code=sm_$(COMPUTE_CAPABILITY) -I $(START_PATH)/cuda-common -I $(START_PATH)/common -o BFS.o -O2 -c $(START_PATH)/bfs/BFS.cu $(PARAMETERS)

libSHOCCommon.a: $(START_PATH)/common/Option.o $(START_PATH)/common/OptionParser.o
	echo "\033[93m[i]\033[0m Compiling dependencies... 🔨"
	ar cr $(START_PATH)/common/libSHOCCommon.a $(START_PATH)/common/OptionParser.o $(START_PATH)/common/Option.o
	ranlib $(START_PATH)/common/libSHOCCommon.a
	echo "\033[92m[✓]\033[0m Done compiling dependencies! 🎉"

clean:
	echo "\033[93m[i]\033[0m Cleaning up... 🗑️"
	$(RM) *.o
	$(RM) $(START_PATH)/common/*.o
	$(RM) $(START_PATH)/common/libSHOCCommon.a
	$(RM) ./bfs
	echo "\033[92m[✓]\033[0m Done cleaning up! 🎉"
