#include "cudacommon.h"
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "OptionParser.h"
#include "Spmv/util.h"
#include "spmv_kernel.cu"

using namespace std;

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing.
//
// Arguments:
//   op: the options parser / parameter database
//
// Programmer: Lukasz Wesolowski
// Creation: June 21, 2010
// Returns:  nothing
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("iterations", OPT_INT, "100", "Number of SpMV iterations "
                 "per pass");
    op.addOption("mm_filename", OPT_STRING, "random", "Name of file "
                 "which stores the matrix in Matrix Market format");
    op.addOption("maxval", OPT_FLOAT, "10", "Maximum value for random "
                 "matrices");
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Runs sparse matrix vector multiplication on the CPU
//
// Arguements:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element of A
//   rowDelimiters: array of size dim+1 holding indices to rows of A;
//                  last element is the index one past the last
//                  element of A
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of rows/columns in the matrix
//   out: input - buffer of size dim
//        output - result from the spmv calculation
//
// Programmer: Lukasz Wesolowski
// Creation: June 23, 2010
// Returns:
//   nothing directly
//   out indirectly through a pointer
// ****************************************************************************
template <typename floatType>
void spmvCpu(const floatType *val, const int *cols, const int *rowDelimiters,
	     const floatType *vec, int dim, floatType *out)
{
    for (int i=0; i<dim; i++)
    {
        floatType t = 0;
        for (int j = rowDelimiters[i]; j < rowDelimiters[i + 1]; j++)
        {
            int col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of GPU results by comparing to CPU results
//
// Arguments:
//   cpuResults: array holding the CPU result vector
//   gpuResults: array hodling the GPU result vector
//   size: number of elements per vector
//   pass: optional iteration number
//
// Programmer: Lukasz Wesolowski
// Creation: June 23, 2010
// Returns:
//   nothing
//   prints "Passed" if the vectors agree within a relative error of
//   MAX_RELATIVE_ERROR and "FAILED" if they are different
// ****************************************************************************
template <typename floatType>
bool verifyResults(const floatType *cpuResults, const floatType *gpuResults,
                   const int size, const int pass = -1)
{
    bool passed = true;
    for (int i = 0; i < size; i++)
    {
        if (fabs(cpuResults[i] - gpuResults[i]) / cpuResults[i]
            > MAX_RELATIVE_ERROR)
        {
//            cout << "Mismatch at i: "<< i << " ref: " << cpuResults[i] <<
//                " dev: " << gpuResults[i] << endl;
            passed = false;
            cerr << "Error: incorrect computed result." << endl;
        }
    }
    if (pass != -1)
    {
        //cout << "Pass "<<pass<<": ";
    }
    if (passed)
    {
        //cout << "Passed" << endl;
    }
    else
    {
        cout << "---FAILED---" << endl;
        cerr << "Error: incorrect computed result." << endl;
    }
    return passed;
}

template <typename floatType, typename texReader>
void csrTest(OptionParser& op, floatType* h_val,
        int* h_cols, int* h_rowDelimiters, floatType* h_vec, floatType* h_out,
        int numRows, int numNonZeroes, floatType* refOut, bool padded)
{
    // Device data structures
    floatType *d_val, *d_vec, *d_out;
    int *d_cols, *d_rowDelimiters;

    // Allocate device memory
    CUDA_SAFE_CALL(cudaMalloc(&d_val,  numNonZeroes * sizeof(floatType)));
    CUDA_SAFE_CALL(cudaMalloc(&d_cols, numNonZeroes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_vec,  numRows * sizeof(floatType)));
    CUDA_SAFE_CALL(cudaMalloc(&d_out,  numRows * sizeof(floatType)));
    CUDA_SAFE_CALL(cudaMalloc(&d_rowDelimiters, (numRows+1) * sizeof(int)));


    CUDA_SAFE_CALL(cudaMemcpy(d_val, h_val,   numNonZeroes * sizeof(floatType),
            cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_cols, h_cols, numNonZeroes * sizeof(int),
            cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_vec, h_vec, numRows * sizeof(floatType),
                cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_rowDelimiters, h_rowDelimiters,
            (numRows+1) * sizeof(int), cudaMemcpyHostToDevice));

    // Bind texture for position
    string suffix;
    if (sizeof(floatType) == sizeof(float))
    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        CUDA_SAFE_CALL(cudaBindTexture(0, vecTex, d_vec, channelDesc,
                numRows * sizeof(float)));
        suffix = "-SP";
    }
    else {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int2>();
        CUDA_SAFE_CALL(cudaBindTexture(0, vecTexD, d_vec, channelDesc,
                numRows * sizeof(int2)));
        suffix = "-DP";
    }

    int passes = op.getOptionInt("passes");
    int iters  = op.getOptionInt("iterations");
    
    // Results description info
    char atts[TEMP_BUFFER_SIZE];
    sprintf(atts, "%d_elements_%d_rows", numNonZeroes, numRows);
    string prefix = "";
    prefix += (padded) ? "Padded_" : "";
    double gflop = 2 * (double) numNonZeroes / 1e9;
    
    // 0: ellpackr, 1: csr-normal-scalar, 2: csr-padded-scalar, 3: csr-normal-vector, 4: csr-padded-vector
    #if (FORMAT == 1 || FORMAT == 2)
        // Setup thread configuration
        int nBlocksScalar = (int) ceil((floatType) numRows / BLOCK_SIZE);
        
        cout << "CSR Scalar Kernel\n";
        for (int k=0; k<passes; k++)
        {
            // Run Scalar Kernel
            for (int j = 0; j < iters; j++)
            {
                spmv_csr_scalar_kernel<floatType, texReader>
                <<<nBlocksScalar, BLOCK_SIZE>>>
                (d_val, d_cols, d_rowDelimiters,
                    #if TEXTURE_MEMORY == 0
                    d_vec,
                    #endif
                    numRows, d_out);
            }
            CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, numRows * sizeof(floatType),
            cudaMemcpyDeviceToHost));
         
            // Compare reference solution to GPU result
            if (! verifyResults(refOut, h_out, numRows, k))
            {
                return;  // If results don't match, don't report performance
            }
        }
    #else // FORMAT == 3 || FORMAT == 4
        // Setup thread configuration
        int new_block_size = 0;
        if (BLOCK_SIZE < 32) {
            new_block_size = 32;
        } else {
            new_block_size = (int) (ceil((double) BLOCK_SIZE / 32.0) * 32.0);
        }

        int nBlocksVector = (int) ceil((floatType) numRows / (floatType)(new_block_size / WARP_SIZE));

        cout << "CSR Vector Kernel\n";
        for (int k=0; k<passes; k++) {
        
            // Run Vector Kernel
            for (int j = 0; j < iters; j++)
            {
                spmv_csr_vector_kernel<floatType, texReader>
                <<<nBlocksVector, new_block_size>>>
                (d_val, d_cols, d_rowDelimiters,
                    #if TEXTURE_MEMORY == 0
                    d_vec,
                    #endif
                    numRows, d_out);
            }
            CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, numRows * sizeof(floatType),
                    cudaMemcpyDeviceToHost));
            cudaThreadSynchronize();
            // Compare reference solution to GPU result
            if (! verifyResults(refOut, h_out, numRows, k))
            {
                return;  // If results don't match, don't report performance
            }
        }
    #endif
    // Free device memory
    CUDA_SAFE_CALL(cudaFree(d_rowDelimiters));
    CUDA_SAFE_CALL(cudaFree(d_vec));
    CUDA_SAFE_CALL(cudaFree(d_out));
    CUDA_SAFE_CALL(cudaFree(d_val));
    CUDA_SAFE_CALL(cudaFree(d_cols));
    CUDA_SAFE_CALL(cudaUnbindTexture(vecTexD));
    CUDA_SAFE_CALL(cudaUnbindTexture(vecTex));
}

template <typename floatType, typename texReader>
void ellPackTest(OptionParser& op, floatType* h_val,
        int* h_cols, int* h_rowDelimiters, floatType* h_vec, floatType* h_out,
        int numRows, int numNonZeroes, floatType* refOut, bool padded,
        int paddedSize)
{
    int *h_rowLengths;
    CUDA_SAFE_CALL(cudaMallocHost(&h_rowLengths, paddedSize * sizeof(int)));
    int maxrl = 0;
    for (int k=0; k<numRows; k++)
    {
        h_rowLengths[k] = h_rowDelimiters[k+1] - h_rowDelimiters[k];
        if (h_rowLengths[k] > maxrl)
        {
            maxrl = h_rowLengths[k];
        }
    }
    for (int p=numRows; p < paddedSize; p++)
    {
        h_rowLengths[p] = 0;
    }

    // Column major format host data structures
    int cmSize = padded ? paddedSize : numRows;
    floatType *h_valcm;
    CUDA_SAFE_CALL(cudaMallocHost(&h_valcm, maxrl * cmSize * sizeof(floatType)));
    int *h_colscm;
    CUDA_SAFE_CALL(cudaMallocHost(&h_colscm, maxrl * cmSize * sizeof(int)));
    convertToColMajor(h_val, h_cols, numRows, h_rowDelimiters, h_valcm,
                              h_colscm, h_rowLengths, maxrl, padded);

    // Device data structures
    floatType *d_val, *d_vec, *d_out;
    int *d_cols, *d_rowLengths;

    // Allocate device memory
    CUDA_SAFE_CALL(cudaMalloc(&d_val,  maxrl*cmSize * sizeof(floatType)));
    CUDA_SAFE_CALL(cudaMalloc(&d_cols, maxrl*cmSize * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_vec,  numRows * sizeof(floatType)));
    CUDA_SAFE_CALL(cudaMalloc(&d_out,  paddedSize * sizeof(floatType)));
    CUDA_SAFE_CALL(cudaMalloc(&d_rowLengths, cmSize * sizeof(int)));

    // Transfer data to device
    CUDA_SAFE_CALL(cudaMemcpy(d_val, h_valcm, maxrl*cmSize * sizeof(floatType),
            cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_cols, h_colscm, maxrl*cmSize * sizeof(int),
            cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_vec, h_vec, numRows * sizeof(floatType),
            cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_rowLengths, h_rowLengths,
            cmSize * sizeof(int), cudaMemcpyHostToDevice));

    // Bind texture for position
    if (sizeof(floatType) == sizeof(float))
    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        CUDA_SAFE_CALL(cudaBindTexture(0, vecTex, d_vec, channelDesc,
                numRows * sizeof(float)));
    }
    else
    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int2>();
        CUDA_SAFE_CALL(cudaBindTexture(0, vecTexD, d_vec, channelDesc,
                numRows * sizeof(int2)));
    }
    int nBlocks = (int) ceil((floatType) cmSize / BLOCK_SIZE);
    int passes = op.getOptionInt("passes");
    int iters  = op.getOptionInt("iterations");

    for (int k=0; k<passes; k++) {
        for (int j = 0; j < iters; j++) {
            spmv_ellpackr_kernel<floatType, texReader><<<nBlocks, BLOCK_SIZE>>>
                    (d_val, d_cols, d_rowLengths,
                        #if TEXTURE_MEMORY == 0
                        d_vec,
                        #endif
                        cmSize, d_out);
        }

        CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, cmSize * sizeof(floatType),
                cudaMemcpyDeviceToHost));

        // Compare reference solution to GPU result
        if (! verifyResults(refOut, h_out, numRows, k)) {
            return;
        }
    }

    // Free device memory
    CUDA_SAFE_CALL(cudaFree(d_rowLengths));
    CUDA_SAFE_CALL(cudaFree(d_vec));
    CUDA_SAFE_CALL(cudaFree(d_out));
    CUDA_SAFE_CALL(cudaFree(d_val));
    CUDA_SAFE_CALL(cudaFree(d_cols));
    if (sizeof(floatType) == sizeof(double))
    {
        CUDA_SAFE_CALL(cudaUnbindTexture(vecTexD));
    }
    else
    {
        CUDA_SAFE_CALL(cudaUnbindTexture(vecTex));
    }
    CUDA_SAFE_CALL(cudaFreeHost(h_rowLengths));
    CUDA_SAFE_CALL(cudaFreeHost(h_valcm));
    CUDA_SAFE_CALL(cudaFreeHost(h_colscm));
}

// ****************************************************************************
// Function: RunTest
//
// Purpose:
//   Executes a run of the sparse matrix - vector multiplication benchmark
//   in either single or double precision
//
// Arguments:
//   op: the options parser / parameter database
//   nRows: number of rows in generated matrix
//
// Returns:  nothing
//
// Programmer: Lukasz Wesolowski
// Creation: June 21, 2010
//
// Modifications:
//
// ****************************************************************************
template <typename floatType, typename texReader>
void RunTest(OptionParser &op, int nRows=0)
{
    // Host data structures
    // Array of values in the sparse matrix
    floatType *h_val, *h_valPad;
    // Array of column indices for each value in h_val
    int *h_cols, *h_colsPad;
    // Array of indices to the start of each row in h_Val
    int *h_rowDelimiters, *h_rowDelimitersPad;
    // Dense vector and space for dev/cpu reference solution
    floatType *h_vec, *h_out, *refOut;
    // nItems = number of non zero elems
    int nItems, nItemsPadded, numRows;

    // This benchmark either reads in a matrix market input file or
    // generates a random matrix
    string inFileName = op.getOptionString("mm_filename");
    if (inFileName == "random")
    {
        numRows = nRows;
        nItems = numRows * numRows / 100; // 1% of entries will be non-zero
        float maxval = op.getOptionFloat("maxval");
        CUDA_SAFE_CALL(cudaMallocHost(&h_val, nItems * sizeof(floatType)));
        CUDA_SAFE_CALL(cudaMallocHost(&h_cols, nItems * sizeof(int)));
        CUDA_SAFE_CALL(cudaMallocHost(&h_rowDelimiters, (numRows + 1) * sizeof(int)));
        fill(h_val, nItems, maxval);
        initRandomMatrix(h_cols, h_rowDelimiters, nItems, numRows);
    }
    else
    {
        char filename[FIELD_LENGTH];
        strcpy(filename, inFileName.c_str());
        readMatrix(filename, &h_val, &h_cols, &h_rowDelimiters,
                &nItems, &numRows);
    }

    // Set up remaining host data
    CUDA_SAFE_CALL(cudaMallocHost(&h_vec, numRows * sizeof(floatType)));
    refOut = new floatType[numRows];
    CUDA_SAFE_CALL(cudaMallocHost(&h_rowDelimitersPad, (numRows + 1) * sizeof(int)));
    fill(h_vec, numRows, op.getOptionFloat("maxval"));

    // Set up the padded data structures
    int paddedSize = numRows + (PAD_FACTOR - numRows % PAD_FACTOR);
    CUDA_SAFE_CALL(cudaMallocHost(&h_out, paddedSize * sizeof(floatType)));
    convertToPadded(h_val, h_cols, numRows, h_rowDelimiters, &h_valPad,
            &h_colsPad, h_rowDelimitersPad, &nItemsPadded);

    // Compute reference solution
    spmvCpu(h_val, h_cols, h_rowDelimiters, h_vec, numRows, refOut);

    // 0: ellpackr, 1: csr-normal-scalar, 2: csr-padded-scalar, 3: csr-normal-vector, 4: csr-padded-vector
    #if (FORMAT == 1 || FORMAT == 3)
        // Test CSR kernels on normal data
        cout << "CSR Test\n";
        csrTest<floatType, texReader>(op, h_val, h_cols,
                h_rowDelimiters, h_vec, h_out, numRows, nItems, refOut, false);
    #elif (FORMAT == 2 || FORMAT == 4)
        // Test CSR kernels on padded data
        cout << "CSR Test -- Padded Data\n";
        csrTest<floatType, texReader>(op, h_valPad, h_colsPad,
                h_rowDelimitersPad, h_vec, h_out, numRows, nItemsPadded, refOut, true);
    #else
        // FORMAT == 0
        // Test ELLPACKR kernel
        cout << "ELLPACKR Test\n";
        ellPackTest<floatType, texReader>(op, h_val, h_cols,
                h_rowDelimiters, h_vec, h_out, numRows, nItems, refOut, false,
                paddedSize);
    #endif

    delete[] refOut;
    CUDA_SAFE_CALL(cudaFreeHost(h_val));
    CUDA_SAFE_CALL(cudaFreeHost(h_cols));
    CUDA_SAFE_CALL(cudaFreeHost(h_rowDelimiters));
    CUDA_SAFE_CALL(cudaFreeHost(h_vec));
    CUDA_SAFE_CALL(cudaFreeHost(h_out));
    CUDA_SAFE_CALL(cudaFreeHost(h_valPad));
    CUDA_SAFE_CALL(cudaFreeHost(h_colsPad));
    CUDA_SAFE_CALL(cudaFreeHost(h_rowDelimitersPad));
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the sparse matrix - vector multiplication benchmark
//
// Arguments:
//   resultDB: stores results from the benchmark
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Lukasz Wesolowski
// Creation: June 21, 2010
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(OptionParser &op)
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    int probSizes[4] = {1024, 8192, 12288, 16384};
    int sizeClass = op.getOptionInt("size") - 1;

    #if PRECISION == 32
        RunTest<float, texReaderSP>(op, probSizes[sizeClass]);
    #else // PRECISION == 64
        RunTest<double, texReaderDP>(op, probSizes[sizeClass]);
    #endif
}
