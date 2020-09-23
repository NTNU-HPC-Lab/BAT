#include <iostream> // For terminal output
#include <fstream> // For file saving
#include <list>
#include <float.h>
#include "tuner_api.h" // KTT API
#include "ktt_json_saver.hpp" // Custom JSON KTT results saver
#include "md.h"
#include "md_helpers.h"

using namespace std;

int main(int argc, char* argv[]) {
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    string kernelFile = "../../../src/kernels/md/md_kernel_helper.cu";
    string referenceKernelFile = "./reference_kernel.cu";
    string kernelName("md_helper");
    ktt::Tuner auto_tuner(platformIndex, deviceIndex, ktt::ComputeAPI::CUDA);

    // NOTE: Maybe a bug in KTT where this has to be OpenCL to work, even for CUDA?
    auto_tuner.setGlobalSizeType(ktt::GlobalSizeType::OpenCL);

    // Problem sizes from SHOC
    uint problemSizes[4] = { 12288, 24576, 36864, 73728 };
    uint inputProblemSize = 1; // Default to first problem size if no input

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

    size_t nAtom = problemSizes[inputProblemSize - 1];

    const ktt::DimensionVector gridSize(nAtom);
    const ktt::DimensionVector blockSize;

    // Add kernel and reference kernel
    ktt::KernelId kernelId = auto_tuner.addKernelFromFile(kernelFile, kernelName, gridSize, blockSize);
    ktt::KernelId referenceKernelId = auto_tuner.addKernelFromFile(referenceKernelFile, kernelName, gridSize, blockSize);

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
    vector<float> positiond(nAtom * 4);
    for (size_t i = 0; i < positiond4.size(); i += 4) {
        positiond.at(i) = positiond4.at(i).x;
        positiond.at(i + 1) = positiond4.at(i).y;
        positiond.at(i + 2) = positiond4.at(i).z;
        positiond.at(i + 3) = positiond4.at(i).w;
    }

    // Add arguments for kernel
    ktt::ArgumentId forcefId = auto_tuner.addArgumentVector(forcef, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId positionfId = auto_tuner.addArgumentVector(positionf, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId forcedId = auto_tuner.addArgumentVector(forced, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId positiondId = auto_tuner.addArgumentVector(positiond, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId maxNeighborsId = auto_tuner.addArgumentScalar(maxNeighbors);
    ktt::ArgumentId neighborListId = auto_tuner.addArgumentVector(neighborList, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId cutsqId = auto_tuner.addArgumentScalar(cutsq);
    ktt::ArgumentId lj1Id = auto_tuner.addArgumentScalar(lj1);
    ktt::ArgumentId lj2Id = auto_tuner.addArgumentScalar(lj2);
    ktt::ArgumentId nAtomId = auto_tuner.addArgumentScalar(nAtom); // The number of elements

    // Add arguments to kernel and reference kernel
    auto_tuner.setKernelArguments(kernelId, vector<ktt::ArgumentId>{forcefId, positionfId, forcedId, positiondId, maxNeighborsId, neighborListId, cutsqId, lj1Id, lj2Id, nAtomId});
    auto_tuner.setKernelArguments(referenceKernelId, vector<ktt::ArgumentId>{forcefId, positionfId, forcedId, positiondId, maxNeighborsId, neighborListId, cutsqId, lj1Id, lj2Id, nAtomId});

    // Get CUDA properties from device 0
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    
    vector<size_t> block_sizes;

    // Initialize the block sizes array
    for (size_t i = 0; i < properties.maxThreadsPerBlock; i++) {
        block_sizes.push_back(i + 1);
    }

    // Add parameters to tune
    auto_tuner.addParameter(kernelId, "BLOCK_SIZE", block_sizes);
    auto_tuner.addParameter(kernelId, "PRECISION", {32, 64});
    auto_tuner.addParameter(kernelId, "TEXTURE_MEMORY", {0}); // KTT does not support texture memory, so this is always disabled
    auto_tuner.addParameter(kernelId, "WORK_PER_THREAD", {1, 2, 3, 4, 5});

    // To set the different block sizes (local size) multiplied by the base (1)
    auto_tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK_SIZE", ktt::ModifierAction::Multiply);
    // To set the different grid sizes (global size) divided by the amount of work per thread
    auto_tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, "WORK_PER_THREAD", ktt::ModifierAction::Divide);

    // Set reference kernel for correctness verification and compare to the computed result
    // NOTE: Due to not being able to specify the precision to match the tuned kernel, this has only been used when developing and testing this benchmark
    // Remember to set the correct precision in the reference kernel when using the reference kernel
    // auto_tuner.setReferenceKernel(kernelId, referenceKernelId, vector<ktt::ParameterPair>{}, vector<ktt::ArgumentId>{forcefId, forcedId});

    // Set the tuner to print in nanoseconds
    auto_tuner.setPrintingTimeUnit(ktt::TimeUnit::Nanoseconds);

    // Tune the kernel
    auto_tuner.tuneKernel(kernelId);

    // Get the best computed result and save it as a JSON to file
    saveJSONFileFromKTTResults(auto_tuner.getBestComputationResult(kernelId), "best-" + kernelName + "-results.json");

    // Print the results to cout and save it as a CSV file
    auto_tuner.printResult(kernelId, cout, ktt::PrintFormat::Verbose);
    auto_tuner.printResult(kernelId, kernelName + "-results.csv", ktt::PrintFormat::CSV);

    return 0;
}