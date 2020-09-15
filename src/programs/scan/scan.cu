#include "cudacommon.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cassert>
#include <iostream>
#include <vector>

#include "OptionParser.h"
#include "scan.h"
#include "scan_kernel.h"

using namespace std;


// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("iterations", OPT_INT, "256", "specify scan iterations");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the scan (parallel prefix sum) benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
// 5/18/2011 - KS - Changing to a non-recursive algorithm
// ****************************************************************************
void
RunBenchmark(OptionParser &op) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    #if PRECISION == 32
        cout << "Running single precision test" << endl;
        RunTest<float, float4>("Scan", op);
    #else
        cout << "Running double precision test" << endl;
        RunTest<double, double4>("Scan-DP", op);
    #endif
}

template <class T, class vecT>
void RunTest(string testName, OptionParser &op)
{
    int probSizes[4] = { 1, 8, 32, 64 };

    int size = probSizes[op.getOptionInt("size")-1];

    // Convert to MiB
    size = (size * 1024 * 1024) / sizeof(T);
    
    // create input data on CPU
    unsigned int bytes = size * sizeof(T);
    cout << size << endl;
    cout << bytes << endl;

    // Allocate Host Memory
    T* h_idata;
    T* reference;
    T* h_odata;
    CUDA_SAFE_CALL(cudaMallocHost((void**) &h_idata,   bytes));
    CUDA_SAFE_CALL(cudaMallocHost((void**) &reference, bytes));
    CUDA_SAFE_CALL(cudaMallocHost((void**) &h_odata,   bytes));

    // Initialize host memory
    cout << "Initializing host memory." << endl;
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = i % 2; // Fill with some pattern
        h_odata[i] = -1;
    }

    // Thread configuration
    // Note: changing this may require updating the kernel calls below
    int num_blocks  = GRID_SIZE;//BLOCK_SIZE/GRID_SIZE_PART;
    int num_threads = BLOCK_SIZE;

    int smem_size = sizeof(T) * num_threads;

    // Allocate device memory
    T* d_idata, *d_odata, *d_block_sums;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_idata, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_odata, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_block_sums, num_blocks * sizeof(T)));

    // Copy data to GPU
    cout << "Copying data to device." << endl;
    CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    int passes = op.getOptionInt("passes");
    int iters = op.getOptionInt("iterations");

    cout << "Running benchmark with size " << size << endl;
    for (int k = 0; k < passes; k++) {
        for (int j = 0; j < iters; j++) {
            // For scan, we use a reduce-then-scan approach

            // Each thread block gets an equal portion of the
            // input array, and computes the sum.
            reduce<T, BLOCK_SIZE><<<num_blocks, num_threads, smem_size>>>
                (d_idata, d_block_sums, size);

            // Next, a top-level exclusive scan is performed on the array
            // of block sums
            scan_single_block<T, BLOCK_SIZE><<<1, num_threads, smem_size*2>>>
                (d_block_sums, num_blocks);

            // Finally, a bottom-level scan is performed by each block
            // that is seeded with the scanned value in block sums
            bottom_scan<T, vecT, BLOCK_SIZE><<<num_blocks, num_threads, 2*smem_size>>>
                (d_idata, d_odata, d_block_sums, size);
        }
        CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata, bytes,
                cudaMemcpyDeviceToHost));

        // If results aren't correct, don't report perf numbers
        if (! scanCPU<T>(h_idata, reference, h_odata, size))
        {
            return;
        }
    }
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_odata));
    CUDA_SAFE_CALL(cudaFree(d_block_sums));
    CUDA_SAFE_CALL(cudaFreeHost(h_idata));
    CUDA_SAFE_CALL(cudaFreeHost(h_odata));
    CUDA_SAFE_CALL(cudaFreeHost(reference));
}


// ****************************************************************************
// Function: scanCPU
//
// Purpose:
//   Simple cpu scan routine to verify device results
//
// Arguments:
//   data : the input data
//   reference : space for the cpu solution
//   dev_result : result from the device
//   size : number of elements
//
// Returns:  nothing, prints relevant info to stdout
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
template <class T>
bool scanCPU(T *data, T* reference, T* dev_result, const size_t size)
{

    bool passed = true;
    T last = 0.0f;

    for (unsigned int i = 0; i < size; ++i)
    {
        reference[i] = data[i] + last;
        last = reference[i];
    }
    for (unsigned int i = 0; i < size; ++i)
    {
        if (reference[i] != dev_result[i])
        {
#ifdef VERBOSE_OUTPUT
            cout << "Mismatch at i: " << i << " ref: " << reference[i]
                 << " dev: " << dev_result[i] << endl;
#endif
            passed = false;
        }
    }
    cout << "Test ";
    if (passed)
        cout << "Passed" << endl;
    else
        cerr << "---FAILED---" << endl;
    return passed;
}
