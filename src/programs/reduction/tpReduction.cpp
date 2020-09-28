#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "cudacommon.h"

// When using MPICH and MPICH-derived MPI implementations, there is a
// naming conflict between stdio.h and MPI's C++ binding.
// Since we do not use the C++ MPI binding, we can avoid the ordering
// issue by ignoring the C++ MPI binding headers.
// This #define should be quietly ignored when using other MPI implementations.
#define MPICH_SKIP_MPICXX
#include "mpi.h"

#include "OptionParser.h"
#include "TPReduction.h"

using namespace std;

// Forward declarations
template <class T>
void RunTest(string test_name, OptionParser &op);

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
template <class T>
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
void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("iterations", OPT_INT, "256", "specify reduction iterations");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Driver for the true parallel reduction benchmark.  Detects double
//   precision capability and calls the RunTest function appropriately.
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: March 1, 2011
//
// Modifications:
//
// ****************************************************************************
void
RunBenchmark(OptionParser &op)
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    #if PRECISION == 32
        cout << "Running single precision test" << endl;
        RunTest<float>("AllReduce-SP", op);
    #elif PRECISION == 64
        cout << "Running double precision test" << endl;
        RunTest<double>("AllReduce-DP", op);
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
template <class T>
void RunTest(string test_name, OptionParser &op)
{

    int comm_size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Per rank size
    int prob_sizes[4] = { 1, 8, 64, 128 };

    int size = prob_sizes[op.getOptionInt("size") - 1];
    size = size * 1024 * 1024 / sizeof(T);

    T* h_idata;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&h_idata, size * sizeof(T)));

    // Initialize host memory
    if (rank == 0)
    {
        cout << "Initializing host memory." << endl;
    }
    for(int i = 0; i < size; i++)
    {
        h_idata[i] = i % 2; //Fill with some pattern
    }

    // allocate device memory
    T* d_idata;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_idata, size * sizeof(T)));

    int num_threads = BLOCK_SIZE;
    int num_blocks = GRID_SIZE;
    int smem_size = num_threads * sizeof(T);

    // allocate mem for the result on host side
    T* h_odata;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&h_odata, num_blocks * sizeof(T)));

    T* d_odata;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_odata, num_blocks * sizeof(T)));

    int passes = op.getOptionInt("passes");
    int iters  = op.getOptionInt("iterations");

    if (rank == 0)
    {
        cout << "Running benchmark.\n";
    }

    for (int k = 0; k < passes; k++)
    {
        double pcie_time=0.0, kernel_time=0.0, mpi_time=0.0;

        MPI_Barrier(MPI_COMM_WORLD);

        cudaTextureObject_t idataTextureObject = 0;

        // Repeatedly transfer input data to GPU and measure average time
        for (int m = 0; m < iters; m++)
        {
            CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata, size * sizeof(T), cudaMemcpyHostToDevice));

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
        }

        // Execute reduction kernel on GPU
        for (int m = 0; m < iters; m++)
        {
            RunTestLaunchKernel<T>(num_blocks, num_threads, smem_size, d_idata, idataTextureObject, d_odata, size);
        }

        // Copy output data back to CPU
        for (int m = 0; m < iters; m++)
        {
            // Copy back to host
            CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata, num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
        }

        T local_result=0, global_result=0;

        // Perform reduction of block sums and MPI allreduce call
        for (int m = 0; m < iters; m++)
        {
            local_result = 0.0f;

            for (int i=0; i<num_blocks; i++)
            {
                local_result += h_odata[i];
            }
            global_result = 0.0f;
            globalReduction(&local_result, &global_result);
        }

        // Compute local reference solution
        T cpu_result = reduceCPU<T>(h_idata, size);
        // Use some error threshold for floating point rounding
        double threshold = 1.0e-6;
        T diff = fabs(local_result - cpu_result);

        if (diff > threshold)
        {
            cout << "Error in local reduction detected in rank " << rank << "\n";
            cout << "Diff: " << diff << endl;
            cerr << "Error: incorrect computed result." << endl;
        }

        if (global_result != (comm_size * local_result))
        {
            cout << "Test Failed, error in global all reduce detected in rank " << rank << endl;
            cerr << "Error: incorrect computed result." << endl;
        }
        else
        {
            if (rank == 0)
            {
                cout << "Test Passed.\n";
            }
        }

        // Calculate results
        char atts[1024];
        sprintf(atts, "%d_itemsPerRank",size);
        double local_gbytes = (double)(size*sizeof(T))/(1000.*1000.*1000.);
        double global_gbytes = local_gbytes * comm_size;
    }
    CUDA_SAFE_CALL(cudaFreeHost(h_idata));
    CUDA_SAFE_CALL(cudaFreeHost(h_odata));
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_odata));
}
