#include <iostream>
#include <fstream>
#include <iterator>
#include <math.h>
#include <limits.h>
#include <cuda_runtime_api.h>
#include "tuner_api.h" // KTT API
#include "ktt_json_saver.hpp" // Custom JSON KTT results saver
#include "Graph.h"

using namespace std;

int main(int argc, char* argv[]) {
    // Tune BFS kernel
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    string kernelFile = "../../../src/kernels/bfs/BFS_kernel.cu";
    string referenceKernelFile = "./reference_kernel.cu";
    string kernelName("BFS_kernel_warp");
    ktt::Tuner auto_tuner(platformIndex, deviceIndex, ktt::ComputeAPI::CUDA);

    // NOTE: Maybe a bug in KTT where this has to be OpenCL to work, even for CUDA
    auto_tuner.setGlobalSizeType(ktt::GlobalSizeType::OpenCL);

    // Problem sizes used in the SHOC benchmark
    uint problemSizes[5] = {1000,10000,100000,1000000,10000000};
    uint inputProblemSize = 1; // Default to the first problem size

    // If only one extra argument and the flag is set for size
    if (argc == 2 && (string(argv[1]) == "--size" || string(argv[1]) == "-s")) {
        cerr << "Error: You need to specify an integer for the problem size." << endl;
        exit(1);
    }

    // If more than two extra arguments and flag is set for size
    if (argc > 2 && (string(argv[1]) == "--size" || string(argv[1]) == "-s")) {
        try {
            inputProblemSize = stoi(argv[2]);

            // Ensure the input problem size is between 1 and 4
            if (inputProblemSize < 1 || inputProblemSize > 4) {
                cerr << "Error: The problem size needs to be an integer in the range 1 to 4." << endl;
                exit(1);
            }
        } catch (const invalid_argument &error) {
            cerr << "Error: You need to specify an integer for the problem size." << endl;
            exit(1);
        }
    }

    unsigned int numVertsGraph = problemSizes[inputProblemSize-1];
    int size = problemSizes[inputProblemSize-1];
    int avg_degree = 2;

    Graph *G=new Graph();

    //Generate simple tree
    G->GenerateSimpleKWayGraph(numVertsGraph, avg_degree);    
    unsigned int *edgeArray=G->GetEdgeOffsets();
    unsigned int *edgeArrayAux=G->GetEdgeList();
    unsigned int adj_list_length=G->GetAdjacencyListLength();
    unsigned int numVerts = G->GetNumVertices();
    vector<int> edgeArrayVector(edgeArray, edgeArray + numVerts+1);
    vector<int> edgeArrayAuxVector(edgeArrayAux, edgeArrayAux + adj_list_length);

    vector<int> costArray(numVerts);
    for (int index = 0; index < numVerts; index++) {
        costArray[index]=UINT_MAX;
    }
    costArray[0]=0;
    vector<int> flag = {0};
    int numVertsInt = numVerts;

    const ktt::DimensionVector gridSize(numVerts);
    const ktt::DimensionVector blockSize;
    const ktt::DimensionVector blockSizeReference(512);

    // Add kernel and reference kernel
    ktt::KernelId kernelId = auto_tuner.addKernelFromFile(kernelFile, kernelName, gridSize, blockSize);
    ktt::KernelId referenceKernelId = auto_tuner.addKernelFromFile(referenceKernelFile, kernelName, gridSize, blockSizeReference);

    // Add arguments for kernel
    // Arrays (A, B, C) have random numbers similar to the program. The numbers are in the range [0, 10)
    // <x>f are floats and <x>d are double values
    ktt::ArgumentId costArrayId = auto_tuner.addArgumentVector(costArray, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId edgeArrayVectorId = auto_tuner.addArgumentVector(edgeArrayVector, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId edgeArrayAuxVectorId = auto_tuner.addArgumentVector(edgeArrayAuxVector, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId W_SZId = auto_tuner.addArgumentScalar(32);
    ktt::ArgumentId numVerticesId = auto_tuner.addArgumentScalar(numVertsInt);
    ktt::ArgumentId currId = auto_tuner.addArgumentScalar(0);
    ktt::ArgumentId flagId = auto_tuner.addArgumentVector(flag, ktt::ArgumentAccessType::ReadWrite);

    // Add arguments to kernel and reference kernel
    auto_tuner.setKernelArguments(kernelId, vector<ktt::ArgumentId>{
        costArrayId, edgeArrayVectorId, edgeArrayAuxVectorId, W_SZId, numVerticesId, currId, flagId});
    auto_tuner.setKernelArguments(referenceKernelId, vector<ktt::ArgumentId>{
        costArrayId, edgeArrayVectorId, edgeArrayAuxVectorId, W_SZId, numVerticesId, currId, flagId});

    // Get the maximum threads per block 
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    size_t maxThreads = min(deviceProp.maxThreadsPerBlock, size);
    // Define a vector of block sizes from 1 to maximum threads per block
    vector<size_t> block_sizes = {};
    for(int i = 1; i < (maxThreads+1); i++) {
        block_sizes.push_back(i);
    }

    // Add parameters to tune
    auto_tuner.addParameter(kernelId, "BLOCK_SIZE", block_sizes);
    auto_tuner.addParameter(kernelId, "UNROLL_OUTER_LOOP", {0, 1});
    auto_tuner.addParameter(kernelId, "UNROLL_INNER_LOOP", {0, 1});
    auto_tuner.addParameter(kernelId, "CHUNK_FACTOR", {1, 2, 4, 8});

    // Multiply block size base (1) by BLOCK_SIZE parameter value
    auto_tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK_SIZE", ktt::ModifierAction::Multiply);
    // Divide total size by block size and chunk factor (and multiply with block size again (because KTT will divide) to make sure number of threads is rounded up)
    auto globalModifier = [](const size_t size, const std::vector<size_t>& vector) {
        return int(ceil(double(size) / double(vector.at(0)) / double(vector.at(1)))) * vector.at(0);
    };
    auto_tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, 
        std::vector<std::string>{"BLOCK_SIZE", "CHUNK_FACTOR"}, globalModifier);

    // Set reference kernel for correctness verification and compare to the computed result
    auto_tuner.setReferenceKernel(kernelId, referenceKernelId, vector<ktt::ParameterPair>{}, vector<ktt::ArgumentId>{costArrayId, flagId});

    // Set the tuner to print in nanoseconds
    auto_tuner.setPrintingTimeUnit(ktt::TimeUnit::Nanoseconds);

    // Tune the kernel
    auto_tuner.tuneKernel(kernelId);

    // Get the best computed result and save it as a JSON to file
    saveJSONFileFromKTTResults(auto_tuner.getBestComputationResult(kernelId), "best-" + kernelName + "-results.json", inputProblemSize);

    // Print the results to cout and save it as a CSV file
    auto_tuner.printResult(kernelId, cout, ktt::PrintFormat::Verbose);
    auto_tuner.printResult(kernelId, kernelName + "-results.csv", ktt::PrintFormat::CSV);

    return 0;
}