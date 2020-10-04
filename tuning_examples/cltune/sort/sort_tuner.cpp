#include <iostream>
#include <fstream>
#include <iterator>
#include <math.h>
#include <cuda_runtime_api.h>
#include "cltune.h" // CLTune API
#include "cltune_json_saver.hpp" // Custom JSON CLTune results saver

using namespace std;

const uint SORT_BLOCK_SIZE = 128;
const uint SCAN_BLOCK_SIZE = 256;
// TODO: remove ^ ?
// Problem sizes from SHOC
uint problemSizes[4] = { 1, 8, 48, 96 };
uint inputProblemSize = 1; // Default to first problem size if no input
string tuningTechnique = "";
int size;
string kernelFile = "../../../src/kernels/sort/sort_kernel_helper.cu";
string referenceKernelFile = "./reference_kernel.cu";
string dataDirectory = "../../../src/kernels/sort/data/";

// Constraint for block sizes and data sizes
// Expects input in order:
// {"SCAN_DATA_SIZE", "SORT_DATA_SIZE", "SCAN_BLOCK_SIZE", "SORT_BLOCK_SIZE"}
auto dataSizeBlockSizeConstraint = [](const vector<size_t> &parameters) {
    return parameters.at(2) / parameters.at(3) == parameters.at(1) / parameters.at(0);
};

void tuneRadixSortBlocks() {
    // Tuning "radixSortBlocks" kernel
    string kernelName("radixSortBlocks");
    // Set the tuning kernel to run on device id 0 and platform 0
    cltune::Tuner auto_tuner(0, 0);

    const size_t radixGlobalWorkSize = size;

    // Add kernel
    size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {radixGlobalWorkSize}, {1});

    // Add parameter for kernel
    auto_tuner.AddParameter(kernel_id, "LOOP_UNROLL_LSB", {0, 1});
    auto_tuner.AddParameter(kernel_id, "INLINE_LSB", {0, 1});
    auto_tuner.AddParameter(kernel_id, "INLINE_SCAN", {0, 1});
    
    auto_tuner.AddParameter(kernel_id, "SCAN_DATA_SIZE", {2});
    auto_tuner.AddParameter(kernel_id, "SORT_DATA_SIZE", {2, 4, 8});
    auto_tuner.DivGlobalSize(kernel_id, {"SORT_DATA_SIZE"});
    
    auto_tuner.AddParameter(kernel_id, "SCAN_BLOCK_SIZE", {256});
    auto_tuner.AddParameter(kernel_id, "SORT_BLOCK_SIZE", {16, 32, 64, 128, 256, 512, 1024});
    auto_tuner.MulLocalSize(kernel_id, {"SORT_BLOCK_SIZE"});

    // Constraint for block sizes and data sizes
    auto_tuner.AddConstraint(
        kernel_id,
        dataSizeBlockSizeConstraint,
        {"SCAN_DATA_SIZE", "SORT_DATA_SIZE", "SCAN_BLOCK_SIZE", "SORT_BLOCK_SIZE"}
    );

    // Set reference kernel for correctness verification and compare to the computed result
    auto_tuner.SetReference({referenceKernelFile}, kernelName, {radixGlobalWorkSize / 4}, {128});
    auto_tuner.AddParameterReference("SCAN_DATA_SIZE", 2);
    auto_tuner.AddParameterReference("SORT_DATA_SIZE", 4);
    auto_tuner.AddParameterReference("SCAN_BLOCK_SIZE", 256);
    auto_tuner.AddParameterReference("SORT_BLOCK_SIZE", 128);

    vector<int> keysIn(size);
    vector<int> valuesIn(size);
    vector<int> keysOut(size); // Also called tempKeys
    vector<int> valuesOut(size); // Also called tempVals

    // Initialize start values for input
    for (int i = 0; i < size; i++) {
        keysIn[i] = valuesIn[i] = i % 1024;
    }

    // Add arguments for kernel
    auto_tuner.AddArgumentScalar(4); // nbits
    auto_tuner.AddArgumentScalar(0); // startbit
    auto_tuner.AddArgumentOutput(keysOut);
    auto_tuner.AddArgumentOutput(valuesOut);
    auto_tuner.AddArgumentInput(keysIn);
    auto_tuner.AddArgumentInput(valuesIn);

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
}

void tuneScan() {
    // Tuning "scan" kernel
    string kernelName("scan_helper");
    // Set the tuning kernel to run on device id 0 and platform 0
    cltune::Tuner auto_tuner(0, 0);

    const size_t reorderFindGlobalWorkSize = size / 2;
    const size_t reorderBlocks = reorderFindGlobalWorkSize / SCAN_BLOCK_SIZE;
    int numElements = 16 * reorderBlocks;
    // unsigned int numBlocks = max(1, (int) ceil((float) numElements / (4.f * SCAN_BLOCK_SIZE)));
    // unsigned int numBlocks = max(1, (int) ceil((float) numElements / (SORT_DATA_SIZE * SCAN_BLOCK_SIZE)));
    // unsigned int numBlocks = max(1, (int) ceil((float) numElements / (4 * SCAN_BLOCK_SIZE)));
    // unsigned int numBlocks = ceil((float) numElements / (4 * SCAN_BLOCK_SIZE));
    unsigned int numBlocks = ceil((float) numElements / (4 * SCAN_BLOCK_SIZE));
    // unsigned int numBlocks = ceil((float) numElements / (SORT_DATA_SIZE * SCAN_BLOCK_SIZE));

    // Add kernel
    // size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {numBlocks}, {SCAN_BLOCK_SIZE});
    // size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {numBlocks}, {1});
    size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {(size_t) numElements}, {1});

    // Add parameter for kernel
    auto_tuner.AddParameter(kernel_id, "LOOP_UNROLL_LOCAL_MEMORY", {0, 1});
    auto_tuner.AddParameter(kernel_id, "INLINE_LOCAL_MEMORY", {0, 1});
    auto_tuner.AddParameter(kernel_id, "SCAN_DATA_SIZE", {2, 4, 8});
    auto_tuner.AddParameter(kernel_id, "SORT_DATA_SIZE", {4});
    auto_tuner.AddParameter(kernel_id, "SCAN_BLOCK_SIZE", {SCAN_BLOCK_SIZE});
    // auto_tuner.AddParameter(kernel_id, "SCAN_BLOCK_SIZE", {16, 32, 64, 128, 256, 512, 1024});
    auto_tuner.MulLocalSize(kernel_id, {"SCAN_BLOCK_SIZE"});
    auto_tuner.AddParameter(kernel_id, "SORT_BLOCK_SIZE", {128});

    // auto_tuner.DivGlobalSize(kernel_id, {"SORT_DATA_SIZE", "SCAN_BLOCK_SIZE"});
    auto_tuner.DivGlobalSize(kernel_id, {"SORT_DATA_SIZE"});

    // Constraint for block sizes and data sizes
    auto_tuner.AddConstraint(
        kernel_id,
        dataSizeBlockSizeConstraint,
        {"SCAN_DATA_SIZE", "SORT_DATA_SIZE", "SCAN_BLOCK_SIZE", "SORT_BLOCK_SIZE"}
    );

    // Get CUDA properties from device 0 
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    int available_shared_memory = properties.sharedMemPerBlock;

    // Constraint for shared memory used
    auto sharedMemoryConstraint = [&](const std::vector<size_t>& vector) {
        return ((vector.at(0) * vector.at(1) * 4 * 2) + (4 * 16 * 2)) <= available_shared_memory;
    };
    auto_tuner.AddConstraint(kernel_id, sharedMemoryConstraint, {"SCAN_BLOCK_SIZE", "SCAN_DATA_SIZE"});

    vector<int> scanOutput(0);
    vector<int> scanInput(0);
    vector<int> blockSums(0);

    // Read input data from files
    ifstream scanOutputFile(dataDirectory + "scan/" + to_string(inputProblemSize) + "-scanOutput");
    ifstream scanInputFile(dataDirectory + "scan/" + to_string(inputProblemSize) + "-scanInput");
    ifstream blockSumsFile(dataDirectory + "scan/" + to_string(inputProblemSize) + "-blockSums");

    // Initialize scanOutput data
    for (istream_iterator<int> it(scanOutputFile), end; it != end; ++it) {
        scanOutput.push_back(*it);
    }

    // Initialize scanInput data
    for (istream_iterator<int> it(scanInputFile), end; it != end; ++it) {
        scanInput.push_back(*it);
    }

    // Initialize blockSums data
    for (istream_iterator<int> it(blockSumsFile), end; it != end; ++it) {
        blockSums.push_back(*it);
    }

    // Add arguments for kernel
    auto_tuner.AddArgumentOutput(scanOutput);
    auto_tuner.AddArgumentInput(scanInput);
    auto_tuner.AddArgumentInput(blockSums);
    auto_tuner.AddArgumentScalar(size);
    auto_tuner.AddArgumentScalar(1); // fullBlock = true
    auto_tuner.AddArgumentScalar(1); // storeSum = true

    // Set reference kernel for correctness verification and compare to the computed result
    auto_tuner.SetReference({referenceKernelFile}, kernelName, {numBlocks}, {256});
    auto_tuner.AddParameterReference("SCAN_DATA_SIZE", 2);
    auto_tuner.AddParameterReference("SORT_DATA_SIZE", 4);
    auto_tuner.AddParameterReference("SCAN_BLOCK_SIZE", 256);
    auto_tuner.AddParameterReference("SORT_BLOCK_SIZE", 128);

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
    saveJSONFileFromCLTuneResults(auto_tuner.GetBestResult(), "best-scan-results.json", inputProblemSize, tuningTechnique);

    // Print the results to cout and save it as a JSON file
    auto_tuner.PrintToScreen();
    auto_tuner.PrintJSON(kernelName + "-results.json", {{"sample", kernelName}});
}

void tuneVectorAddUniform4() {
    // Tuning "vectorAddUniform4" kernel
    string kernelName("vectorAddUniform4");
    // Set the tuning kernel to run on device id 0 and platform 0
    cltune::Tuner auto_tuner(0, 0);

    const size_t reorderFindGlobalWorkSize = size / 2;
    const size_t reorderBlocks = reorderFindGlobalWorkSize / SCAN_BLOCK_SIZE;
    int numElements = 16 * reorderBlocks;
    // unsigned int numBlocks = max(1, (unsigned int) ceil((float) numElements / (SORT_DATA_SIZE * SCAN_BLOCK_SIZE)));
    unsigned int numBlocks = max(1, (int) ceil((float) numElements / (4.f * SCAN_BLOCK_SIZE)));
    // TODO: update this numBlocks: 4.0f?
    // TODO: update others over here

    // Add kernel
    size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {numBlocks}, {SCAN_BLOCK_SIZE});

    // Add parameter for kernel
    auto_tuner.AddParameter(kernel_id, "LOOP_UNROLL_ADD_UNIFORM", {0, 1});
    auto_tuner.AddParameter(kernel_id, "SCAN_BLOCK_SIZE", {SCAN_BLOCK_SIZE});
    auto_tuner.AddParameter(kernel_id, "SORT_BLOCK_SIZE", {SORT_BLOCK_SIZE});
    auto_tuner.AddParameter(kernel_id, "SORT_DATA_SIZE", {2});
    auto_tuner.AddParameter(kernel_id, "SCAN_DATA_SIZE", {4});

    vector<int> scanOutput(0);
    vector<int> blockSums(0);

    // Read input data from files
    ifstream scanOutputFile(dataDirectory + "uniform/" + to_string(inputProblemSize) + "-scanOutput");
    ifstream blockSumsFile(dataDirectory + "uniform/" + to_string(inputProblemSize) + "-blockSums");

    // Initialize scanOutput data
    for (istream_iterator<int> it(scanOutputFile), end; it != end; ++it) {
        scanOutput.push_back(*it);
    }

    // Initialize blockSums data
    for (istream_iterator<int> it(blockSumsFile), end; it != end; ++it) {
        blockSums.push_back(*it);
    }

    // Add arguments for kernel
    auto_tuner.AddArgumentOutput(scanOutput);
    auto_tuner.AddArgumentInput(blockSums);
    auto_tuner.AddArgumentScalar(size);

    // Set reference kernel for correctness verification and compare to the computed result
    auto_tuner.SetReference({referenceKernelFile}, kernelName, {numBlocks}, {SCAN_BLOCK_SIZE});
    // TODO: update reference kernel launching over

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
}

int main(int argc, char* argv[]) {
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

    // Convert to size to MB
    size = (problemSizes[inputProblemSize - 1] * 1024 * 1024) / sizeof(uint);

    // Tune all kernels
    // tuneRadixSortBlocks();
    tuneScan();
    // tuneVectorAddUniform4();
    
    return 0;
}
