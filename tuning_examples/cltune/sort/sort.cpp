#include <iostream>
#include <fstream>
#include <iterator>
#include <math.h>
#include "cltune.h" // CLTune API

using namespace std;

// Problem sizes from SHOC
const uint SORT_BLOCK_SIZE = 128;
const uint SCAN_BLOCK_SIZE = 256;
uint problemSizes[4] = { 1, 8, 48, 96 };
uint inputProblemSize = 1; // Default to first problem size if no input
int size;
string kernelFile = "../../../src/kernels/sort/sort_kernel.cu";
string referenceKernelFile = "./reference_kernel.cu";
string dataDirectory = "../../../src/kernels/sort/data/";

void tuneRadixSortBlocks() {
    // Tuning "radixSortBlocks" kernel
    string kernelName("radixSortBlocks");
    // Set the tuning kernel to run on device id 0 and platform 0
    cltune::Tuner auto_tuner(0, 0);

    const size_t radixGlobalWorkSize = size / 4;

    // Add kernel
    size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {radixGlobalWorkSize}, {SORT_BLOCK_SIZE});

    // Add parameter for kernel
    auto_tuner.AddParameter(kernel_id, "LOOP_UNROLL_LSB", {0, 1});

    // Set reference kernel for correctness verification and compare to the computed result
    auto_tuner.SetReference({referenceKernelFile}, kernelName, {radixGlobalWorkSize}, {SORT_BLOCK_SIZE});

    vector<int> keysIn(size);
    vector<int> valuesIn(size);
    vector<int> keysOut(size); // Also called tempKeys
    vector<int> valuesOut(size); // Also called tempVals

    // Initialize start values for input
    for (int i = 0; i < size; i++) {
        keysIn[i] = valuesIn[i] = i % 1024;
    }

    // Add arguments for kernel
    auto_tuner.AddArgumentScalar(4);
    auto_tuner.AddArgumentScalar(0);
    auto_tuner.AddArgumentOutput(keysIn);
    auto_tuner.AddArgumentOutput(valuesOut);
    auto_tuner.AddArgumentInput(keysIn);
    auto_tuner.AddArgumentInput(valuesIn);

    auto_tuner.Tune();

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
    unsigned int numBlocks = max(1, (int) ceil((float) numElements / (4.f * SCAN_BLOCK_SIZE)));

    // Add kernel
    size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {numBlocks}, {SCAN_BLOCK_SIZE});

    // Add parameter for kernel
    auto_tuner.AddParameter(kernel_id, "LOOP_UNROLL_ADD_UNIFORM", {0, 1});

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

    auto_tuner.Tune();

    // Print the results to cout and save it as a JSON file
    auto_tuner.PrintToScreen();
    auto_tuner.PrintJSON(kernelName + "-results.json", {{"sample", kernelName}});
}

void tuneScan() {
    // Tuning "scan" kernel
    string kernelName("scan");
    // Set the tuning kernel to run on device id 0 and platform 0
    cltune::Tuner auto_tuner(0, 0);

    const size_t reorderFindGlobalWorkSize = size / 2;
    const size_t reorderBlocks = reorderFindGlobalWorkSize / SCAN_BLOCK_SIZE;
    int numElements = 16 * reorderBlocks;
    unsigned int numBlocks = max(1, (int) ceil((float) numElements / (4.f * SCAN_BLOCK_SIZE)));

    // Add kernel
    size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {numBlocks}, {SCAN_BLOCK_SIZE});

    // Add parameter for kernel
    auto_tuner.AddParameter(kernel_id, "LOOP_UNROLL_LOCAL_MEMORY", {0, 1});

    vector<int> scanOutput(0);
    vector<int> scanInput(0);
    vector<int> blockSums(0);

    // Read input data from files
    ifstream scanOutputFile(dataDirectory + kernelName + "/" + to_string(inputProblemSize) + "-scanOutput");
    ifstream scanInputFile(dataDirectory + kernelName + "/" + to_string(inputProblemSize) + "-scanOutput");
    ifstream blockSumsFile(dataDirectory + kernelName + "/" + to_string(inputProblemSize) + "-blockSums");

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
    auto_tuner.AddArgumentScalar(numElements);
    auto_tuner.AddArgumentScalar(1); // fullBlock = true
    auto_tuner.AddArgumentScalar(1); // storeSum = true

    // Set reference kernel for correctness verification and compare to the computed result
    auto_tuner.SetReference({referenceKernelFile}, kernelName, {numBlocks}, {SCAN_BLOCK_SIZE});

    auto_tuner.Tune();

    // Print the results to cout and save it as a JSON file
    auto_tuner.PrintToScreen();
    auto_tuner.PrintJSON(kernelName + "-results.json", {{"sample", kernelName}});
}

int main(int argc, char* argv[]) {
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
    size = (problemSizes[inputProblemSize - 1] * 1024 * 1024) / sizeof(uint);

    // Tune all kernels
    tuneRadixSortBlocks();
    tuneScan();
    tuneVectorAddUniform4();
    
    return 0;
}
