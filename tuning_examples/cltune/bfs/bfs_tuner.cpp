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

    string tuningTechnique = "";

    
    // If only one extra argument and the flag is set for tuning technique
    if (argc == 2 && (string(argv[1]) == "--technique" || string(argv[1]) == "-t")) {
        cerr << "Error: You need to specify a tuning technique." << endl;
        exit(1);
    }

    // Check if the provided arguments does not match in size
    if ((argc - 1) % 2 != 0) {
        cerr << "Error: You need to specify correct number of input arguments." << endl;
        exit(1);
    }

    // Loop arguments and add if found
    for (int i = 1; i < argc; i++) {
        // Skip the argument value iterations
        if (i % 2 == 0) {
            continue;
        }

        // Check for problem size
        if (string(argv[i]) == "--size" || string(argv[i]) == "-s") {
            try {
                inputProblemSize = stoi(argv[i + 1]);

                // Ensure the input problem size is between 1 and 4
                if (inputProblemSize < 1 || inputProblemSize > 4) {
                    cerr << "Error: The problem size needs to be an integer in the range 1 to 4." << endl;
                    exit(1);
                }
            } catch (const invalid_argument &error) {
                cerr << "Error: You need to specify an integer for the problem size." << endl;
                exit(1);
            }
        // Check for tuning technique
        } else if (string(argv[i]) == "--technique" || string(argv[i]) == "-t") {
            tuningTechnique = argv[i + 1];
        } else {
            cerr << "Error: Unsupported argument " << "`" << argv[i] << "`" << endl;
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
    int size = problemSizes[inputProblemSize-1];
    int avg_degree = 2;

    Graph *G=new Graph();

    //Generate simple tree
    G->GenerateSimpleKWayGraph(numVertsGraph,avg_degree);    
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

    // Get the maximum threads per block 
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    unsigned int maxThreads = min(deviceProp.maxThreadsPerBlock, size);
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
    auto_tuner.SetReference({referenceKernelFile}, kernelName, {numVerts}, {512});

    // Add arguments for kernel
    auto_tuner.AddArgumentOutput(costArray);
    auto_tuner.AddArgumentInput(edgeArrayVector);
    auto_tuner.AddArgumentInput(edgeArrayAuxVector);
    auto_tuner.AddArgumentScalar(32);
    auto_tuner.AddArgumentScalar(numVertsInt);
    auto_tuner.AddArgumentScalar(0);
    auto_tuner.AddArgumentOutput(flag);

    // Use 50% of the total search space
    double searchFraction = 0.5;

    // Select the tuning technique for this benchmark
    if (tuningTechnique == "annealing") {
        double maxTemperature = 4.0f;
        auto_tuner.UseAnnealing(searchFraction, {maxTemperature});
    } else if (tuningTechnique == "pso") {
        double swarmSize = 4.0f;
        auto_tuner.UsePSO(searchFraction, swarmSize, 0.4, 0.0, 0.4);
    } else if (tuningTechnique == "random") {
        auto_tuner.UseRandomSearch(searchFraction);
    } else if (tuningTechnique == "brute_force") {
        auto_tuner.UseFullSearch();
    } else {
        cerr << "Error: Unsupported tuning technique: `" << tuningTechnique << "`." << endl;
        exit(1);
    }

    auto_tuner.Tune();

    // Get the best computed result and save it as a JSON to file
    saveJSONFileFromCLTuneResults(auto_tuner.GetBestResult(), "best-" + kernelName + "-results.json", inputProblemSize, tuningTechnique);

    // Print the results to cout and save it as a JSON file
    auto_tuner.PrintToScreen();
    auto_tuner.PrintJSON(kernelName + "-results.json", {{"sample", kernelName}});
    
    return 0;
}
