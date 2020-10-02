#include <iostream>
#include <fstream>
#include <iterator>
#include <math.h>
#include <limits.h>
#include <cuda_runtime_api.h>
#include "tuner_api.h" // KTT API
#include "ktt_json_saver.hpp" // Custom JSON KTT results saver
#include "Spmv/util.h"

using namespace std;

int main(int argc, char* argv[]) {
    // Tune BFS kernel
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    string kernelFile = "../../../src/kernels/spmv/spmv_kernel.cu";
    string referenceKernelFile = "./reference_kernel.cu";
    string kernelName("spmv_kernel");
    ktt::Tuner auto_tuner(platformIndex, deviceIndex, ktt::ComputeAPI::CUDA);

    // NOTE: Maybe a bug in KTT where this has to be OpenCL to work, even for CUDA
    auto_tuner.setGlobalSizeType(ktt::GlobalSizeType::OpenCL);

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

    const ktt::DimensionVector gridSize(numRows);
    const ktt::DimensionVector blockSize;
    const ktt::DimensionVector blockSizeReference(128);

    // Add kernel and reference kernel
    ktt::KernelId kernelId = auto_tuner.addKernelFromFile(kernelFile, kernelName, gridSize, blockSize);
    ktt::KernelId referenceKernelId = auto_tuner.addKernelFromFile(referenceKernelFile, kernelName, gridSize, blockSizeReference);
    
    // Use the refernence kernel below instead of the one above if the format is CSR Vector
    /*
    int csr_vector_total_size = numRows * 32;
    const ktt::DimensionVector gridSizeSCRVector(csr_vector_total_size);
    ktt::KernelId referenceKernelId = auto_tuner.addKernelFromFile(referenceKernelFile, kernelName, gridSizeSCRVector, blockSizeReference);
    */

    // Add arguments for kernel
    ktt::ArgumentId val_csr_sp_v_ID = auto_tuner.addArgumentVector(val_csr_sp_v, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId val_csr_dp_v_ID = auto_tuner.addArgumentVector(val_csr_dp_v, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId val_csr_sp_pad_v_ID = auto_tuner.addArgumentVector(val_csr_sp_pad_v, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId val_csr_dp_pad_v_ID = auto_tuner.addArgumentVector(val_csr_dp_pad_v, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId val_ellpackr_sp_v_ID = auto_tuner.addArgumentVector(val_ellpackr_sp_v, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId val_ellpackr_dp_v_ID = auto_tuner.addArgumentVector(val_ellpackr_dp_v, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId cols_csr_ID = auto_tuner.addArgumentVector(cols_csr, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId cols_csr_pad_ID = auto_tuner.addArgumentVector(cols_csr_pad, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId cols_ellpackr_v_ID = auto_tuner.addArgumentVector(cols_ellpackr_v, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId rowDelimiters_csr_ID = auto_tuner.addArgumentVector(rowDelimiters_csr, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId rowDelimiters_csr_pad_ID = auto_tuner.addArgumentVector(rowDelimiters_csr_pad, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId rowLengths_ellpackr_ID = auto_tuner.addArgumentVector(rowLengths_ellpackr, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId vec_csr_sp_ID = auto_tuner.addArgumentVector(vec_csr_sp, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId vec_csr_dp_ID = auto_tuner.addArgumentVector(vec_csr_dp, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId numRowsID = auto_tuner.addArgumentScalar(numRows);
    ktt::ArgumentId nItemsPaddedID = auto_tuner.addArgumentScalar(nItemsPadded);
    ktt::ArgumentId out_sp_ID = auto_tuner.addArgumentVector(out_sp, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId out_dp_ID = auto_tuner.addArgumentVector(out_dp, ktt::ArgumentAccessType::ReadWrite);

    // Add arguments to kernel and reference kernel
    auto_tuner.setKernelArguments(kernelId, vector<ktt::ArgumentId>{
        val_csr_sp_v_ID, val_csr_dp_v_ID, val_csr_sp_pad_v_ID, val_csr_dp_pad_v_ID,
        val_ellpackr_sp_v_ID, val_ellpackr_dp_v_ID, cols_csr_ID, cols_csr_pad_ID,
        cols_ellpackr_v_ID, rowDelimiters_csr_ID, rowDelimiters_csr_pad_ID, rowLengths_ellpackr_ID,
        vec_csr_sp_ID, vec_csr_dp_ID, numRowsID, nItemsPaddedID, 
        out_sp_ID, out_dp_ID});
    auto_tuner.setKernelArguments(referenceKernelId, vector<ktt::ArgumentId>{
        val_csr_sp_v_ID, val_csr_dp_v_ID, val_csr_sp_pad_v_ID, val_csr_dp_pad_v_ID,
        val_ellpackr_sp_v_ID, val_ellpackr_dp_v_ID, cols_csr_ID, cols_csr_pad_ID,
        cols_ellpackr_v_ID, rowDelimiters_csr_ID, rowDelimiters_csr_pad_ID, rowLengths_ellpackr_ID,
        vec_csr_sp_ID, vec_csr_dp_ID, numRowsID, nItemsPaddedID, 
        out_sp_ID, out_dp_ID});

    // Get the maximum threads per block 
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    unsigned int maxThreads = deviceProp.maxThreadsPerBlock;
    // Define a vector of block sizes from 1 to maximum threads per block
    vector<long unsigned int> block_sizes = {};
    for(int i = 1; i < (maxThreads+1); i++) {
        block_sizes.push_back(i);
    }

    // Add parameters to tune
    auto_tuner.addParameter(kernelId, "BLOCK_SIZE", block_sizes);
    auto_tuner.addParameter(kernelId, "PRECISION", {32, 64});
    // Formats: 0: ellpackr, 1: csr-normal-scalar, 2:  csr-padded-scalar, 3: csr-normal-vector, 4: csr-padded-vector
    auto_tuner.addParameter(kernelId, "FORMAT", {3});//, 1, 2, 3, 4});
    auto_tuner.addParameter(kernelId, "UNROLL_LOOP_1", {0, 1});
    auto_tuner.addParameter(kernelId, "UNROLL_LOOP_2", {0, 1});
    auto_tuner.addParameter(kernelId, "THREADS_PER_ROW", {1, 32});
    auto_tuner.addParameter(kernelId, "NOT_TEXTURE_MEMORY", {1});

    // Add constraint for only using block sizes that are a multiple of 32 for CSR vector format (format 3 or 4)
    auto blockSizeLimit = [] (std::vector<size_t> v) {
        return (v[1] < 3 || v[0] % 32 == 0);
    };
    auto_tuner.addConstraint(kernelId, {"BLOCK_SIZE", "FORMAT"}, blockSizeLimit);

    // Add constraint for only unrolling loop 2 when the format is CSR Vector
    auto unrollLoop2 = [] (std::vector<size_t> v) {
        return (v[0] > 2 || v[1] < 1);
    };
    auto_tuner.addConstraint(kernelId, {"FORMAT", "UNROLL_LOOP_2"}, unrollLoop2);
    
    // Add a constraint for what to multiply the total thread size by. If the format is the CSR vector format (3 or 4)
    // the total number of threads should be multiplied by 32.
    auto threadMultiply = [] (std::vector<size_t> v) {
        if (v[1] > 2) {
            return v[0] == 32;
        } else {
            return v[0] == 1;
        }
    };
    auto_tuner.addConstraint(kernelId, {"THREADS_PER_ROW", "FORMAT"}, threadMultiply);

    // Multiply block size base (1) by BLOCK_SIZE parameter value
    auto_tuner.setThreadModifier(kernelId, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK_SIZE", ktt::ModifierAction::Multiply);
    // Multiply grid size base (total size) by THREADS_PER_ROW parameter value    
    auto globalModifier = [](const size_t size, const std::vector<size_t>& vector) {
        return int(ceil(double(size) * double(vector.at(1)) / double(vector.at(0)))) * vector.at(0);
    };
    auto_tuner.setThreadModifier(kernelId, ktt::ModifierType::Global, ktt::ModifierDimension::X, 
        std::vector<std::string>{"BLOCK_SIZE", "THREADS_PER_ROW"}, globalModifier);

    // Set reference kernel for correctness verification and compare to the computed result
    // auto_tuner.setReferenceKernel(kernelId, referenceKernelId, vector<ktt::ParameterPair>{}, vector<ktt::ArgumentId>{out_sp_ID, out_dp_ID});

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