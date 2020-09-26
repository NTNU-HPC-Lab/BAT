#include <iostream>
#include <fstream>
#include <iterator>
#include <math.h>
#include <cuda_runtime_api.h>
#include <list>
#include <float.h>
#include "cltune.h" // CLTune API
#include "cltune_json_saver.hpp" // Custom JSON CLTune results saver
#include "md.h"
#include "md_helpers.h"

using namespace std;

int main(int argc, char* argv[]) {
    // Problem sizes used in the SHOC benchmark
    uint problemSizes[4] = { 12288, 24576, 36864, 73728 };
    uint inputProblemSize = 1; // Default to the first problem size if no input

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

    string kernelFile = "../../../src/kernels/md/md_kernel_helper.cu";
    string referenceKernelFile = "./reference_kernel.cu";

    // Tune "MD" kernel
    string kernelName("md_helper");
    // Set the tuning kernel to run on device id 0 and platform 0
    cltune::Tuner auto_tuner(0, 0);

    size_t nAtom = problemSizes[inputProblemSize - 1];

    // Add kernel
    size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {nAtom}, {1});

    // Get CUDA properties from device 0
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    
    vector<size_t> block_sizes;

    // Initialize the block sizes array
    for (size_t i = 0; i < properties.maxThreadsPerBlock; i++) {
        block_sizes.push_back(i + 1);
    }

    // Add parameters to tune
    auto_tuner.AddParameter(kernel_id, "BLOCK_SIZE", block_sizes);
    // To set the different block sizes (local size) multiplied by the base (1)
    auto_tuner.MulLocalSize(kernel_id, {"BLOCK_SIZE"});

    auto_tuner.AddParameter(kernel_id, "PRECISION", {32, 64});
    auto_tuner.AddParameter(kernel_id, "TEXTURE_MEMORY", {0}); // CLTune does not support texture memory, so this is always disabled
    auto_tuner.AddParameter(kernel_id, "WORK_PER_THREAD", {1, 2, 3, 4, 5});
    // To set the different grid sizes (global size) divided by the amount of work per thread
    auto_tuner.DivGlobalSize(kernel_id, {"WORK_PER_THREAD"});

    // Set reference kernel for correctness verification and compare to the computed result
    // NOTE: Due to not being able to specify the precision to match the tuned kernel, this has only been used when developing and testing this benchmark
    // Remember to set the correct precision in the reference kernel when using the reference kernel
    // auto_tuner.SetReference({referenceKernelFile}, kernelName, {nAtom}, {1});

    // Use the same seed for random number as in the SHOC benchmark
    srand48(8650341L);

    vector<float> forcef(nAtom * 3);
    vector<double> forced(nAtom * 3);
    vector<float4> positionf4(nAtom);
    vector<double4> positiond4(nAtom);
    vector<int> neighborList(nAtom * maxNeighbors);

    // Initialize start values for input
    for (int i = 0; i < nAtom; i++) {
        positiond4[i].x = drand48() * domainEdge;
        positionf4[i].x = (float) positiond4[i].x;
        positiond4[i].y = drand48() * domainEdge;
        positionf4[i].y = (float)positiond4[i].y;
        positiond4[i].z = drand48() * domainEdge;
        positionf4[i].z = (float)positiond4[i].z;
    }

    // Build the neighbor lists for both precisions
    buildNeighborList<float, float4>(nAtom, &positionf4[0], &neighborList[0]);
    buildNeighborList<double, double4>(nAtom, &positiond4[0], &neighborList[0]);
    
    // Copy back the float data
    vector<float> positionf(nAtom * 4);
    for (size_t i = 0; i < positionf4.size(); i += 4) {
        positionf.at(i) = positionf4.at(i).x;
        positionf.at(i + 1) = positionf4.at(i).y;
        positionf.at(i + 2) = positionf4.at(i).z;
        positionf.at(i + 3) = positionf4.at(i).w;
    }

    // Copy back the double data
    vector<double> positiond(nAtom * 4);
    for (size_t i = 0; i < positiond4.size(); i += 4) {
        positiond.at(i) = positiond4.at(i).x;
        positiond.at(i + 1) = positiond4.at(i).y;
        positiond.at(i + 2) = positiond4.at(i).z;
        positiond.at(i + 3) = positiond4.at(i).w;
    }

    // Add arguments for kernel
    auto_tuner.AddArgumentOutput(forcef);
    auto_tuner.AddArgumentInput(positionf);
    auto_tuner.AddArgumentOutput(forced);
    auto_tuner.AddArgumentInput(positiond);
    auto_tuner.AddArgumentScalar(maxNeighbors);
    auto_tuner.AddArgumentInput(neighborList);
    auto_tuner.AddArgumentScalar(cutsq);
    auto_tuner.AddArgumentScalar(lj1);
    auto_tuner.AddArgumentScalar(lj2);
    auto_tuner.AddArgumentScalar(nAtom); // The number of elements

    auto_tuner.Tune();

    // Get the best computed result and save it as a JSON to file
    saveJSONFileFromCLTuneResults(auto_tuner.GetBestResult(), "best-" + kernelName + "-results.json");

    // Print the results to cout and save it as a JSON file
    auto_tuner.PrintToScreen();
    auto_tuner.PrintJSON(kernelName + "-results.json", {{"sample", kernelName}});

    return 0;
}
