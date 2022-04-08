#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern "C" {

#include "triad_kernel.cu"

using namespace std;

// Select which precision that are used in the calculations
#if PRECISION == 32
    #define DATA_TYPE float
#elif PRECISION == 64
    #define DATA_TYPE double
#endif

// ****************************************************************************
// Originated from the SHOC benchmark
// Function: RunBenchmark
//
// Purpose:
//   Implements the Stream Triad benchmark in CUDA.  This benchmark
//   is designed to test CUDA's overall data transfer speed. It executes
//   a vector addition operation with no temporal reuse. Data is read
//   directly from the global memory. This implementation tiles the input
//   array and pipelines the vector addition computation with
//   the data download for the next tile. However, since data transfer from
//   host to device is much more expensive than the simple vector computation,
//   data transfer operations should completely dominate the execution time.
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser (contains input parameters)
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************

// Return the time used in float to help choosing the best configuration
float triad_host() {
    // TODO implement verbose?
    const bool verbose = false;
    const int n_passes = 5;

    // 256k through 8M bytes
    const int nSizes = 9;
    const size_t blockSizes[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
    const size_t memSize = 16384;
    const size_t numMaxFloats = 1024 * memSize / 4;
    const size_t halfNumFloats = numMaxFloats / 2;

    // Create some host memory pattern
    srand48(8650341L);
    DATA_TYPE *h_mem;
    cudaMallocHost((void**) &h_mem, sizeof(DATA_TYPE) * numMaxFloats);

    // Allocate some device memory
    DATA_TYPE* d_memA0, *d_memB0, *d_memC0;
    cudaMalloc((void**) &d_memA0, blockSizes[nSizes - 1] * 1024);
    cudaMalloc((void**) &d_memB0, blockSizes[nSizes - 1] * 1024);
    cudaMalloc((void**) &d_memC0, blockSizes[nSizes - 1] * 1024);

    DATA_TYPE* d_memA1, *d_memB1, *d_memC1;
    cudaMalloc((void**) &d_memA1, blockSizes[nSizes - 1] * 1024);
    cudaMalloc((void**) &d_memB1, blockSizes[nSizes - 1] * 1024);
    cudaMalloc((void**) &d_memC1, blockSizes[nSizes - 1] * 1024);

    DATA_TYPE scalar = 1.75f;

    const size_t blockSize = BLOCK_SIZE;

    // For measuring the time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalElapsedTime = 0.0;

    // Number of passes. Use a large number for stress testing.
    // A small value is sufficient for computing sustained performance.
    char sizeStr[256];
    for (int pass = 0; pass < n_passes; ++pass)
    {
        // Step through sizes forward
        for (int i = 0; i < nSizes; ++i)
        {
            int elemsInBlock = blockSizes[i] * 1024 / sizeof(DATA_TYPE);
            for (int j = 0; j < halfNumFloats; ++j) {
                h_mem[j] = h_mem[halfNumFloats + j] = (DATA_TYPE) (drand48() * 10.0);
            }

            // Copy input memory to the device
            if (verbose)
                cout << ">> Executing Triad with vectors of length "
                << numMaxFloats << " and block size of "
                << elemsInBlock << " elements." << "\n";
            sprintf(sizeStr, "Block:%05ldKB", blockSizes[i]);

            // start submitting blocks of data of size elemsInBlock
            // overlap the computation of one block with the data
            // download for the next block and the results upload for
            // the previous block
            int crtIdx = 0;
            size_t globalWorkSize = ceil((double)elemsInBlock / (double)blockSize / (double) WORK_PER_THREAD);

            cudaStream_t streams[2];
            cudaStreamCreate(&streams[0]);
            cudaStreamCreate(&streams[1]);

            // Start the timing
            cudaEventRecord(start, 0);

            cudaMemcpyAsync(d_memA0, h_mem, blockSizes[i] * 1024, cudaMemcpyHostToDevice, streams[0]);
            cudaMemcpyAsync(d_memB0, h_mem, blockSizes[i] * 1024, cudaMemcpyHostToDevice, streams[0]);

            triad<<<globalWorkSize, blockSize, 0, streams[0]>>>
                    (d_memA0, d_memB0, d_memC0, scalar, elemsInBlock);

            if (elemsInBlock < numMaxFloats)
            {
                // start downloading data for next block
                cudaMemcpyAsync(d_memA1, h_mem + elemsInBlock, blockSizes[i] * 1024, cudaMemcpyHostToDevice, streams[1]);
                cudaMemcpyAsync(d_memB1, h_mem + elemsInBlock, blockSizes[i] * 1024, cudaMemcpyHostToDevice, streams[1]);
            }

            int blockIdx = 1;
            unsigned int currStream = 1;
            while (crtIdx < numMaxFloats)
            {
                currStream = blockIdx & 1;
                // Start copying back the answer from the last kernel
                if (currStream)
                {
                    cudaMemcpyAsync(h_mem + crtIdx, d_memC0, elemsInBlock * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost, streams[0]);
                }
                else
                {
                    cudaMemcpyAsync(h_mem + crtIdx, d_memC1, elemsInBlock * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost, streams[1]);
                }

                crtIdx += elemsInBlock;

                if (crtIdx < numMaxFloats)
                {
                    // Execute the kernel
                    if (currStream)
                    {
                        triad<<<globalWorkSize, blockSize, 0, streams[1]>>>
                                (d_memA1, d_memB1, d_memC1, scalar, elemsInBlock);
                    }
                    else
                    {
                        triad<<<globalWorkSize, blockSize, 0, streams[0]>>>
                                (d_memA0, d_memB0, d_memC0, scalar, elemsInBlock);
                    }
                }

                if (crtIdx+elemsInBlock < numMaxFloats)
                {
                    // Download data for next block
                    if (currStream)
                    {
                        cudaMemcpyAsync(d_memA0, h_mem+crtIdx+elemsInBlock, blockSizes[i]*1024, cudaMemcpyHostToDevice, streams[0]);
                        cudaMemcpyAsync(d_memB0, h_mem+crtIdx+elemsInBlock, blockSizes[i]*1024, cudaMemcpyHostToDevice, streams[0]);
                    }
                    else
                    {
                        cudaMemcpyAsync(d_memA1, h_mem+crtIdx+elemsInBlock, blockSizes[i]*1024, cudaMemcpyHostToDevice, streams[1]);
                        cudaMemcpyAsync(d_memB1, h_mem+crtIdx+elemsInBlock, blockSizes[i]*1024, cudaMemcpyHostToDevice, streams[1]);
                    }
                }
                blockIdx += 1;
                currStream = !currStream;
            }

            cudaDeviceSynchronize();
            // Stop the events and save elapsed time
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);
            totalElapsedTime += elapsedTime;

            // Checking memory for correctness. The two halves of the array
            // should have the same results.
            if (verbose) cout << ">> checking memory\n";
            for (int j=0; j<halfNumFloats; ++j)
            {
                if (h_mem[j] != h_mem[j+halfNumFloats])
                {
                    cout << "Error; hostMem[" << j << "]=" << h_mem[j]
                         << " is different from its twin element hostMem["
                         << (j+halfNumFloats) << "]: "
                         << h_mem[j+halfNumFloats] << "stopping check\n";
                    cerr << "Error: incorrect computed result." << endl;
                    throw "Correctness verification failed";
                }
            }
            if (verbose) cout << ">> finish!" << endl;

            // Zero out the test host memory
            for (int j=0; j<numMaxFloats; ++j) {
                h_mem[j] = 0.0f;
            }
        }
    }

    // Cleanup
    cudaFree(d_memA0);
    cudaFree(d_memB0);
    cudaFree(d_memC0);
    cudaFree(d_memA1);
    cudaFree(d_memB1);
    cudaFree(d_memC1);
    cudaFreeHost(h_mem);

    return totalElapsedTime;
}
}
