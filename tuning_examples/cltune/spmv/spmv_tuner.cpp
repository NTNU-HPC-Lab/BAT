#include <iostream>
#include <fstream>
#include <iterator>
#include <math.h>
#include <limits.h>
#include "cltune.h" // CLTune API
#include "cltune_json_saver.hpp" // Custom JSON CLTune results saver
#include "Spmv/util.h"

using namespace std;

int main(int argc, char* argv[]) {
    // Problem sizes used in the SHOC benchmark
    uint problemSizes[4] = {1024, 8192, 12288, 16384};
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

    string kernelFile = "../../../src/kernels/spmv/spmv_kernel.cu";
    string referenceKernelFile = "./reference_kernel.cu";

    // Tune "spmv_kernel" kernel
    string kernelName("spmv_kernel");
    // Set the tuning kernel to run on device id 0 and platform 0
    cltune::Tuner auto_tuner(0, 0);

    // Calculating input values and allocating memory 
    int numRows = problemSizes[inputProblemSize-1];
    int nItems = numRows * numRows / 100;
    int paddedSize = numRows + (16 - numRows % 16);
    int nItemsPadded;
    float maxval = 10;
    float *val_csr_sp = (float *) malloc(nItems * sizeof(float)); 
    float *val_csr_pad_sp; 
    double *val_csr_dp = (double *) malloc(nItems * sizeof(double));
    double *val_csr_pad_dp; 

    int *cols = (int *) malloc(nItems * sizeof(int));
    int *cols_pad;
    int *rowDelimiters = (int *) malloc((numRows + 1) * sizeof(int));
    int *rowDelimiters_pad = (int *) malloc((numRows + 1) * sizeof(int));
    float *vec_sp = (float *) malloc(numRows * sizeof(float));
    double *vec_dp = (double *) malloc(numRows * sizeof(double));

    int *rowLengths = (int *) malloc((paddedSize) * sizeof(int));

    fill(val_csr_sp, nItems, maxval);
    fill(val_csr_dp, nItems, maxval);
    initRandomMatrix(cols, rowDelimiters, nItems, numRows);
    fill(vec_sp, numRows, maxval);
    fill(vec_dp, numRows, maxval);
    convertToPadded(val_csr_sp, cols, numRows, rowDelimiters, &val_csr_pad_sp, &cols_pad, rowDelimiters_pad, &nItemsPadded);
    convertToPadded(val_csr_dp, cols, numRows, rowDelimiters, &val_csr_pad_dp, &cols_pad, rowDelimiters_pad, &nItemsPadded);

    int maxrl = 0;
    for (int k = 0; k < numRows; k++) {
        rowLengths[k] = rowDelimiters[k+1] - rowDelimiters[k];
        if (rowLengths[k] > maxrl)
        {
            maxrl = rowLengths[k];
        }
    }
    for (int p = numRows; p < paddedSize; p++) {
        rowLengths[p] = 0;
    }
    float *val_ellpackr_sp = (float *) malloc(maxrl * numRows * sizeof(float));
    double *val_ellpackr_dp = (double *) malloc(maxrl * numRows * sizeof(double));
    int *cols_ellpackr = (int *) malloc(maxrl * numRows * sizeof(int));
    convertToColMajor(val_csr_sp, cols, numRows, rowDelimiters, val_ellpackr_sp, cols_ellpackr, rowLengths, maxrl, false);
    convertToColMajor(val_csr_dp, cols, numRows, rowDelimiters, val_ellpackr_dp, cols_ellpackr, rowLengths, maxrl, false);

    // Converting pointers to vectors for passing them as arguments
    vector<float> val_csr_sp_v(val_csr_sp, val_csr_sp + nItems);
    vector<double> val_csr_dp_v(val_csr_dp, val_csr_dp + nItems);
    vector<float> val_csr_sp_pad_v(val_csr_pad_sp, val_csr_pad_sp + nItemsPadded);
    vector<double> val_csr_dp_pad_v(val_csr_pad_dp, val_csr_pad_dp + nItemsPadded);
    vector<float> val_ellpackr_sp_v(val_ellpackr_sp, val_ellpackr_sp + (maxrl * numRows));
    vector<double> val_ellpackr_dp_v(val_ellpackr_dp, val_ellpackr_dp + (maxrl * numRows));

    vector<int> cols_csr(cols, cols + nItems);
    vector<int> cols_csr_pad(cols_pad, cols_pad + nItemsPadded);
    vector<int> cols_ellpackr_v(cols_ellpackr, cols_ellpackr + (maxrl * numRows));

    vector<int> rowDelimiters_csr(rowDelimiters, rowDelimiters + (numRows + 1));
    vector<int> rowDelimiters_csr_pad(rowDelimiters_pad, rowDelimiters_pad + (numRows + 1));
    vector<int> rowLengths_ellpackr(rowLengths, rowLengths + paddedSize);

    vector<float> vec_csr_sp(vec_sp, vec_sp + numRows);
    vector<float> vec_csr_dp(vec_dp, vec_dp + numRows);

    // Initializing output vectors
    vector<float> out_sp(paddedSize);
    vector<double> out_dp(paddedSize);

    // Get the maximum threads per block 
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    unsigned int maxThreads = deviceProp.maxThreadsPerBlock;
    // Define a vector of block sizes from 1 to maximum threads per block
    vector<long unsigned int> block_sizes = {};
    for(int i = 1; i < (maxThreads+1); i++) {
        block_sizes.push_back(i);
    }

    // Total threads to be launched
    long unsigned int totalSize = numRows;
    // Add kernel
    size_t kernel_id = auto_tuner.AddKernel({kernelFile}, kernelName, {totalSize}, {1});

    // Add parameters to tune
    auto_tuner.AddParameter(kernel_id, "PROBLEM_SIZE", {inputProblemSize});
    auto_tuner.AddParameter(kernel_id, "BLOCK_SIZE", {block_sizes});
    auto_tuner.AddParameter(kernel_id, "PRECISION", {32, 64});
    // Formats: 0: ellpackr, 1: csr-normal-scalar, 2:  csr-padded-scalar, 3: csr-normal-vector, 4: csr-padded-vector
    auto_tuner.AddParameter(kernel_id, "FORMAT", {0, 1, 2, 3, 4});
    auto_tuner.AddParameter(kernel_id, "UNROLL_LOOP_1", {0, 1});
    auto_tuner.AddParameter(kernel_id, "UNROLL_LOOP_2", {0, 1});
    auto_tuner.AddParameter(kernel_id, "THREADS_PER_ROW", {1, 32});
    auto_tuner.AddParameter(kernel_id, "NOT_TEXTURE_MEMORY", {1});

    // Add constraint for only using block sizes that are a multiple of 32 for CSR vector format (format 3 or 4)
    auto BlockSizeLimit = [] (std::vector<size_t> v) {
        return (v[1] < 3 || v[0] % 32 == 0);
    };
    auto_tuner.AddConstraint(kernel_id, BlockSizeLimit, {"BLOCK_SIZE", "FORMAT"});
    
    // Add a constraint for what to multiply the total thread size by. If the format is the CSR vector format (3 or 4)
    // the total number of threads should be multiplied by 32.
    auto k = [] (std::vector<size_t> v) {
        if (v[1] > 2) {
            return v[0] == 32;
        } else {
            return v[0] == 1;
        }
    };
    auto_tuner.AddConstraint(kernel_id, k, {"THREADS_PER_ROW", "FORMAT"});

    // Multiply the base number (1) of threads per block with the parameter value
    auto_tuner.MulLocalSize(kernel_id, {"BLOCK_SIZE"});
    // Multiply the total thread number by 32 if the format is 3 or 4, if not multiply by 1.
    auto_tuner.MulGlobalSize(kernel_id, {"THREADS_PER_ROW"});


    // Set reference kernel for correctness verification and compare to the computed result
    // The reference kernel needs to be commented out when the tuning happens because there can only be one reference kernel,
    // and the kernel gives different output depending on the format.
    /*
    long unsigned int csr_vector_total_size = totalSize * 32;
    auto_tuner.SetReference({referenceKernelFile}, kernelName, {totalSize}, {128});
    auto_tuner.AddParameterReference("FORMAT", 0);
    auto_tuner.AddParameterReference("PRECISION", 32);
    */
    
    // Add arguments for kernel
    auto_tuner.AddArgumentInput(val_csr_sp_v);
    auto_tuner.AddArgumentInput(val_csr_dp_v);
    auto_tuner.AddArgumentInput(val_csr_sp_pad_v);
    auto_tuner.AddArgumentInput(val_csr_dp_pad_v);
    auto_tuner.AddArgumentInput(val_ellpackr_sp_v);
    auto_tuner.AddArgumentInput(val_ellpackr_dp_v);
    auto_tuner.AddArgumentInput(cols_csr);
    auto_tuner.AddArgumentInput(cols_csr_pad);
    auto_tuner.AddArgumentInput(cols_ellpackr_v);
    auto_tuner.AddArgumentInput(rowDelimiters_csr);
    auto_tuner.AddArgumentInput(rowDelimiters_csr_pad);
    auto_tuner.AddArgumentInput(rowLengths_ellpackr);
    auto_tuner.AddArgumentInput(vec_csr_sp);
    auto_tuner.AddArgumentInput(vec_csr_dp);
    auto_tuner.AddArgumentScalar(numRows);
    auto_tuner.AddArgumentScalar(nItemsPadded);
    auto_tuner.AddArgumentOutput(out_sp);
    auto_tuner.AddArgumentOutput(out_dp);

    auto_tuner.Tune();

    // Get the best computed result and save it as a JSON to file
    saveJSONFileFromCLTuneResults(auto_tuner.GetBestResult(), "best-" + kernelName + "-results.json");

    // Print the results to cout and save it as a JSON file
    auto_tuner.PrintToScreen();
    auto_tuner.PrintJSON(kernelName + "-results.json", {{"sample", kernelName}});
    
    return 0;
}
