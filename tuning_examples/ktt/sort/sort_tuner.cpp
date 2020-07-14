#include <iostream> // For terminal output
#include <fstream> // For file saving
#include "tuner_api.h" // KTT API
#include "tunable_sort.h" // To help with the launch of kernels
#include "reference_sort.h" // To check the correctness of the computed kernels
#include "ktt_json_saver.hpp" // Custom JSON KTT results saver

using namespace std;

int main(int argc, char* argv[]) {
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    string kernelFile = "../../../src/kernels/sort/sort_kernel.cu";

    ktt::Tuner auto_tuner(platformIndex, deviceIndex, ktt::ComputeAPI::CUDA);

    // NOTE: Maybe a bug in KTT where this has to be OpenCL to work, even for CUDA?
    auto_tuner.setGlobalSizeType(ktt::GlobalSizeType::OpenCL);

    const ktt::DimensionVector gridSize;
    const ktt::DimensionVector blockSize;

    vector<ktt::KernelId> kernelIds(5);
    kernelIds[0] = auto_tuner.addKernelFromFile(kernelFile, "radixSortBlocks", gridSize, blockSize);
    kernelIds[1] = auto_tuner.addKernelFromFile(kernelFile, "findRadixOffsets", gridSize, blockSize);
    kernelIds[2] = auto_tuner.addKernelFromFile(kernelFile, "reorderData", gridSize, blockSize);
    kernelIds[3] = auto_tuner.addKernelFromFile(kernelFile, "vectorAddUniform4", gridSize, blockSize);
    kernelIds[4] = auto_tuner.addKernelFromFile(kernelFile, "scan", gridSize, blockSize);

    // Problem sizes from SHOC
    uint problemSizes[4] = { 1, 8, 48, 96 };
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

    // Convert to size to MB
    uint size = (problemSizes[inputProblemSize - 1] * 1024 * 1024) / sizeof(uint);
    uint nBits = 4;
    uint startBit = 0;

    vector<uint> keysIn(size);
    vector<uint> valuesIn(size);
    vector<uint> keysOut(size); // Also called tempKeys
    vector<uint> valuesOut(size); // Also called tempVals

    // Initialize start values for input
    for (int i = 0; i < size; i++) {
        keysIn[i] = valuesIn[i] = i % 1024;
    }

    // Add arguments
    ktt::ArgumentId sizeId = auto_tuner.addArgumentScalar(size);
    ktt::ArgumentId nBitsId = auto_tuner.addArgumentScalar(nBits);
    ktt::ArgumentId startBitId = auto_tuner.addArgumentScalar(startBit);

    ktt::ArgumentId keysInId = auto_tuner.addArgumentVector(keysIn, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId valuesInId = auto_tuner.addArgumentVector(valuesIn, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId keysOutId = auto_tuner.addArgumentVector(keysOut, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId valuesOutId = auto_tuner.addArgumentVector(valuesOut, ktt::ArgumentAccessType::ReadWrite);

    ktt::ArgumentId countersId = auto_tuner.addArgumentVector(vector<uint>(1), ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId counterSumsId = auto_tuner.addArgumentVector(vector<uint>(1), ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId blockOffsetsId = auto_tuner.addArgumentVector(vector<uint>(1), ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId reorderFindBlocksId = auto_tuner.addArgumentScalar(0);

    ktt::ArgumentId numberOfElementsId = auto_tuner.addArgumentScalar(0); // Also called numElements
    ktt::ArgumentId blockSumsId = auto_tuner.addArgumentVector(vector<uint>(1), ktt::ArgumentAccessType::ReadWrite);

    ktt::ArgumentId scanInputId = auto_tuner.addArgumentVector(vector<uint>(1), ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId scanOutputId = auto_tuner.addArgumentVector(vector<uint>(1), ktt::ArgumentAccessType::ReadWrite);

    ktt::ArgumentId fullBlockId = auto_tuner.addArgumentScalar(0);
    ktt::ArgumentId storeSumId = auto_tuner.addArgumentScalar(0);

    // Create a composition of the kernels
    ktt::KernelId compositionId = auto_tuner.addComposition("sort", kernelIds, make_unique<TunableSort>(kernelIds,
                                                                                                        size,
                                                                                                        startBitId,
                                                                                                        keysInId,
                                                                                                        valuesInId,
                                                                                                        keysOutId,
                                                                                                        valuesOutId,
                                                                                                        countersId,
                                                                                                        counterSumsId,
                                                                                                        blockOffsetsId,
                                                                                                        reorderFindBlocksId,
                                                                                                        numberOfElementsId,
                                                                                                        blockSumsId,
                                                                                                        scanOutputId,
                                                                                                        scanInputId,
                                                                                                        fullBlockId,
                                                                                                        storeSumId));

    // Add arguments for each composition kernel
    // Add arguments for "radixSortBlocks" kernel
    auto_tuner.setCompositionKernelArguments(compositionId, kernelIds[0], vector<ktt::ArgumentId>{nBitsId, startBitId, keysOutId, valuesOutId, keysInId, valuesInId});
    // Add arguments for "findRadixOffsets" kernel
    auto_tuner.setCompositionKernelArguments(compositionId, kernelIds[1], vector<ktt::ArgumentId>{keysOutId, countersId, blockOffsetsId, startBitId, sizeId, reorderFindBlocksId});
    // Add arguments for "reorderData" kernel
    auto_tuner.setCompositionKernelArguments(compositionId, kernelIds[2], vector<ktt::ArgumentId>{startBitId, keysOutId, valuesOutId, keysInId, valuesInId, blockOffsetsId, counterSumsId, countersId, reorderFindBlocksId});
    // Add arguments for "vectorAddUniform4" kernel
    auto_tuner.setCompositionKernelArguments(compositionId, kernelIds[3], vector<ktt::ArgumentId>{scanOutputId, blockSumsId, numberOfElementsId});
    // Add arguments for "scan" kernel
    auto_tuner.setCompositionKernelArguments(compositionId, kernelIds[4], vector<ktt::ArgumentId>{scanOutputId, scanInputId, blockSumsId, numberOfElementsId, fullBlockId, storeSumId});

    // Add parameters to tune
    auto_tuner.addParameter(compositionId, "LOOP_UNROLL_LSB", {0, 1});
    auto_tuner.addParameter(compositionId, "LOOP_UNROLL_LOCAL_MEMORY", {0, 1});
    auto_tuner.addParameter(compositionId, "LOOP_UNROLL_ADD_UNIFORM", {0, 1});

    // Set reference class for correctness verification and compare to the computed result
    auto_tuner.setReferenceClass(compositionId, make_unique<ReferenceSort>(valuesIn), vector<ktt::ArgumentId>{valuesOutId});

    // Tune the composition of kernels
    auto_tuner.tuneKernel(compositionId);

    // Get the best computed result and save it as a JSON to file
    saveJSONFileFromKTTResults(auto_tuner.getBestComputationResult(compositionId), "best-sort-results.json");

    // Print the results to cout and save it as a CSV file
    auto_tuner.printResult(compositionId, cout, ktt::PrintFormat::Verbose);
    auto_tuner.printResult(compositionId, "sort-results.csv", ktt::PrintFormat::CSV);

    return 0;
}