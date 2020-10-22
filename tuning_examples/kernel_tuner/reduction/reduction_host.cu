#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" {

#include "reduction_kernel.cu"

using namespace std;

// Select which precision that are used in the calculations
// And define the replacements for the template inputs
#if PRECISION == 32
    #define T float
#elif PRECISION == 64
    #define T double
#endif

float RunTest(string testName);

// ****************************************************************************
// Function: reduceCPU
//
// Purpose:
//   Simple cpu reduce routine to verify device results
//
// Arguments:
//   data : the input data
//   size : size of the input data
//
// Returns:  sum of the data
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
T reduceCPU(const T *data, int size)
{
    T sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += data[i];
    }
    return sum;
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Driver for the reduction benchmark.  Detects double precision capability
//   and calls the RunTest function appropriately
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
//
// ****************************************************************************

// Return the time used in float to help choosing the best configuration
float reduction_host() {
    #if PRECISION == 32
        cout << "Running single precision test" << endl;
        return RunTest("Reduction");
    #elif PRECISION == 64
        cout << "Running double precision test" << endl;
        return RunTest("Reduction-DP");
    #endif
}

// ****************************************************************************
// Function: RunTest
//
// Purpose:
//   Primary method for the reduction benchmark
//
// Arguments:
//   testName: the name of the test currently being executed (specifying SP or
//             DP)
//   resultDB: results from the benchmark are stored in this db
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
float RunTest(string testName)
{
    int prob_sizes[4] = { 1, 8, 32, 64 };

    int input_problem_size = PROBLEM_SIZE;
    int size = prob_sizes[input_problem_size - 1];
    size = (size * 1024 * 1024) / sizeof(T);

    T* h_idata;
    cudaMallocHost((void**)&h_idata, size * sizeof(T));

    // Initialize host memory
    cout << "Initializing host memory." << endl;
    for(int i = 0; i < size; i++)
    {
        h_idata[i] = i % 3; //Fill with some pattern
    }

    // allocate device memory
    T* d_idata;
    cudaMalloc((void**)&d_idata, size * sizeof(T));

    // TODO: remove this NB comment???
    int num_threads = BLOCK_SIZE; // NB: Update template to kernel launch if this is changed
    int num_blocks = GRID_SIZE;
    int smem_size = sizeof(T) * num_threads;
    // allocate mem for the result on host side
    T* h_odata;
    cudaMallocHost((void**)&h_odata, num_blocks * sizeof(T));

    T* d_odata;
    cudaMalloc((void**)&d_odata, num_blocks * sizeof(T));

    int passes = 10;
    int iters  = 100;

    // For measuring the time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalElapsedTime = 0.0;

    cout << "Running benchmark." << endl;
    for (int k = 0; k < passes; k++)
    {
        // Copy data to GPU
        cudaMemcpy(d_idata, h_idata, size * sizeof(T), cudaMemcpyHostToDevice);

        cudaTextureObject_t idataTextureObject = 0;
        
    #if TEXTURE_MEMORY
        // Setup the texture memory
        // Create the texture resource descriptor
        cudaResourceDesc resourceDescriptor;
        memset(&resourceDescriptor, 0, sizeof(resourceDescriptor));
        resourceDescriptor.resType = cudaResourceTypeLinear;
        resourceDescriptor.res.linear.devPtr = d_idata;
        #if PRECISION == 32
            resourceDescriptor.res.linear.desc.f = cudaChannelFormatKindFloat;
        #elif PRECISION == 64
            resourceDescriptor.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        #endif
        resourceDescriptor.res.linear.desc.x = 32;
        #if PRECISION == 64
            resourceDescriptor.res.linear.desc.y = 32;
        #endif
        resourceDescriptor.res.linear.sizeInBytes = size * sizeof(T);

        // Create the texture resource descriptor
        cudaTextureDesc textureDescriptor;
        memset(&textureDescriptor, 0, sizeof(textureDescriptor));
        textureDescriptor.readMode = cudaReadModeElementType;

        // Create the texture object
        cudaCreateTextureObject(&idataTextureObject, &resourceDescriptor, &textureDescriptor, NULL);
    #endif

        // Start the timing
        cudaEventRecord(start, 0);

        // Execute kernel
        for (int m = 0; m < iters; m++)
        {
            reduce<<<num_blocks,num_threads, smem_size>>>
                (d_idata, idataTextureObject, d_odata, size);
        }

        // Stop the events and save elapsed time
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        totalElapsedTime += elapsedTime;

        // Copy back to host
        cudaMemcpy(h_odata, d_odata, num_blocks * sizeof(T), cudaMemcpyDeviceToHost);

        T dev_result = 0;
        for (int i=0; i<num_blocks; i++)
        {
            dev_result += h_odata[i];
        }

        // compute reference solution
        T cpu_result = reduceCPU(h_idata, size);
        double threshold = 1.0e-6;
        T diff = fabs(dev_result - cpu_result);

        cout << "Test ";
        if (diff < threshold) {
            cout << "Passed" << endl;
        } else {
            cout << "FAILED" << endl;
            cout << "Diff: " << diff << endl;
            cerr << "Error: incorrect computed result." << endl;
            // (don't report erroneous results)
            throw "Correctness verification failed";
        }
    }
    cudaFreeHost(h_idata);
    cudaFreeHost(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    return totalElapsedTime;
}
}
