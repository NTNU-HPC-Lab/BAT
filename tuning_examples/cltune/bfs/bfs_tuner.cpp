#include <iostream>
#include <fstream>
#include <iterator>
#include <math.h>
#include <limits.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cltune.h" // CLTune API
#include "cltune_json_saver.hpp" // Custom JSON CLTune results saver
#include "Graph.h"

using namespace std;

int main(int argc, char* argv[]) {
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

    string kernelFile = "../../../src/kernels/bfs/BFS_kernel.cu";
    string referenceKernelFile = "./reference_kernel.cu";

    // Tune "BFS_kernel_warp" kernel
    string kernelName("BFS_kernel_warp");
    // Set the tuning kernel to run on device id 0 and platform 0
    cltune::Tuner auto_tuner(0, 0);

    unsigned int numVertsGraph = problemSizes[inputProblemSize-1];
    int avg_degree = 2;

    Graph *G=new Graph();

    //Generate simple tree
    G->GenerateSimpleKWayGraph(numVertsGraph,avg_degree);    
    unsigned int *edgeArray=G->GetEdgeOffsets();
    unsigned int *edgeArrayAux=G->GetEdgeList();
    unsigned int adj_list_length=G->GetAdjacencyListLength();
    unsigned int numVerts = G->GetNumVertices();
    //unsigned int numEdges = G->GetNumEdges();
    vector<int> edgeArrayVector(edgeArray, edgeArray + numVerts+1);
    vector<int> edgeArrayAuxVector(edgeArrayAux, edgeArrayAux + adj_list_length);

    vector<int> costArray(numVerts);
    for (int index = 0; index < numVerts; index++) {
        costArray[index]=UINT_MAX;
    }
    costArray[0]=0;
    vector<int> flag = {0};
    int numVerts2 = numVerts;

    // Get the maximum threads per block 
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    unsigned int maxThreads = deviceProp.maxThreadsPerBlock;
    // Define a vector of block sizes from 1 to maximum threads per block
    vector<long unsigned int> block_sizes = {};
    for(int i = 1; i < (maxThreads+1); i++) {
        block_sizes.push_back(i);
    }

    // Add kernel
    size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {numVerts}, {1});

    // Add parameters to tune
    auto_tuner.AddParameter(kernel_id, "BLOCK_SIZE", block_sizes);
    auto_tuner.AddParameter(kernel_id, "UNROLL_OUTER_LOOP", {0, 1});
    auto_tuner.AddParameter(kernel_id, "UNROLL_INNER_LOOP", {0, 1});
    auto_tuner.AddParameter(kernel_id, "CHUNK_FACTOR", {1, 2, 4, 8});

    // Set the different block sizes (local size) multiplied by the base (1)
    auto_tuner.MulLocalSize(kernel_id, {"BLOCK_SIZE"});
    // Divide the total number of threads by the chunk factor
    auto_tuner.DivGlobalSize(kernel_id, {"CHUNK_FACTOR"});

    // Set reference kernel for correctness verification and compare to the computed result
    auto_tuner.SetReference({referenceKernelFile}, kernelName, {numVerts}, {1024});

    // Add arguments for kernel
    auto_tuner.AddArgumentOutput(costArray);
    auto_tuner.AddArgumentInput(edgeArrayVector);
    auto_tuner.AddArgumentInput(edgeArrayAuxVector);
    auto_tuner.AddArgumentScalar(32);
    auto_tuner.AddArgumentScalar(numVerts2);
    auto_tuner.AddArgumentScalar(0);
    auto_tuner.AddArgumentOutput(flag);

    auto_tuner.Tune();

    // Get the best computed result and save it as a JSON to file
    saveJSONFileFromCLTuneResults(auto_tuner.GetBestResult(), "best-" + kernelName + "-results.json", inputProblemSize);

    // Print the results to cout and save it as a JSON file
    auto_tuner.PrintToScreen();
    auto_tuner.PrintJSON(kernelName + "-results.json", {{"sample", kernelName}});
    
    return 0;
}
