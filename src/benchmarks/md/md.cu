#include <cassert>
#include <cfloat>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <list>
#include <math.h>
#include <stdlib.h>
#include "cudacommon.h"
#include "md.h"
#include "OptionParser.h"
#include "Utility.h"
#include "md_helpers.h"

using namespace std;

// Forward Declaration
template <class T, class forceVecType, class posVecType, bool useTexture, typename texReader>
void runTest(const string& testName, OptionParser& op);

#include "md_kernel.cu"

// ****************************************************************************
// Function: checkResults
//
// Purpose:
//   Check device results against cpu results -- this is the CPU equivalent of
//
// Arguments:
//      d_force:   forces calculated on the device
//      position:  positions of atoms
//      neighList: atom neighbor list
//      nAtom:     number of atoms
// Returns:  true if results match, false otherwise
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ****************************************************************************
#ifdef TEMPLATE
template <class T, class forceVecType, class posVecType>
#endif
bool checkResults(forceVecType* d_force, posVecType *position, int *neighList, int nAtom)
{
    for (int i = 0; i < nAtom; i++)
    {
        posVecType ipos = position[i];
        forceVecType f = {0.0f, 0.0f, 0.0f};
        int j = 0;
        while (j < maxNeighbors)
        {
            int jidx = neighList[j*nAtom + i];
            posVecType jpos = position[jidx];
            // Calculate distance
            T delx = ipos.x - jpos.x;
            T dely = ipos.y - jpos.y;
            T delz = ipos.z - jpos.z;
            T r2inv = delx*delx + dely*dely + delz*delz;

            // If distance is less than cutoff, calculate force
            if (r2inv < cutsq) {

                r2inv = 1.0f/r2inv;
                T r6inv = r2inv * r2inv * r2inv;
                T force = r2inv*r6inv*(lj1*r6inv - lj2);

                f.x += delx * force;
                f.y += dely * force;
                f.z += delz * force;
            }
            j++;
        }
        // Check the results
        T diffx = (d_force[i].x - f.x) / d_force[i].x;
        T diffy = (d_force[i].y - f.y) / d_force[i].y;
        T diffz = (d_force[i].z - f.z) / d_force[i].z;
        T err = sqrt(diffx*diffx) + sqrt(diffy*diffy) + sqrt(diffz*diffz);
        if (err > (3.0 * EPSILON))
        {
            cout << "Test Failed, idx: " << i << " diff: " << err << "\n";
            cout << "f.x: " << f.x << " df.x: " << d_force[i].x << "\n";
            cout << "f.y: " << f.y << " df.y: " << d_force[i].y << "\n";
            cout << "f.z: " << f.z << " df.z: " << d_force[i].z << "\n";
            cout << "Test FAILED\n";
            cerr << "Error: incorrect computed result." << endl;
            return false;
        }
    }
    cout << "Test Passed\n";
    return true;
}


// ********************************************************
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
// ********************************************************
void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("iterations", OPT_INT, "1", "specify MD kernel iterations", 'r');
}

// ********************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the md benchmark
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
// ********************************************************
void
RunBenchmark(OptionParser &op)
{
    #if PRECISION == 32
        cout << "Running single precision test" << endl;
        runTest<float, float3, float4, TEXTURE_MEMORY, texReader_sp>("MD-LJ", op);
    #elif PRECISION == 64
        cout << "Running double precision test" << endl;
        runTest<double, double3, double4, TEXTURE_MEMORY, texReader_dp>("MD-LJ-DP", op);
    #endif
}

#ifdef TEMPLATE
template <class T, class forceVecType, class posVecType, bool useTexture, typename texReader>
#endif
void runTest(const string& testName
#ifdef TEMPLATE
, OptionParser& op
#endif
)
{
    // Problem Parameters
    const int probSizes[4] = { 12288, 24576, 36864, 73728 };
    #ifdef TEMPLATE
    int sizeClass = op.getOptionInt("size");
    #else
    int sizeClass = PROBLEM_SIZE;
    #endif
    assert(sizeClass >= 0 && sizeClass < 5);
    int nAtom = probSizes[sizeClass - 1];

    // Allocate problem data on host
    posVecType*   position;
    forceVecType* force;
    int* neighborList;

    CUDA_SAFE_CALL(cudaMallocHost((void**)&position, nAtom*sizeof(posVecType)));
    CUDA_SAFE_CALL(cudaMallocHost((void**)&force,    nAtom*sizeof(forceVecType)));
    CUDA_SAFE_CALL(cudaMallocHost((void**)&neighborList, nAtom*maxNeighbors*sizeof(int)));

    // Allocate device memory for position and force
    forceVecType* d_force;
    posVecType*   d_position;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_force,    nAtom*sizeof(forceVecType)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_position, nAtom*sizeof(posVecType)));

    // Allocate device memory for neighbor list
    int* d_neighborList;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborList, nAtom*maxNeighbors*sizeof(int)));

    cout << "Initializing test problem (this can take several minutes for large problems)" << endl;

    // Seed random number generator
    srand48(8650341L);

    // Initialize positions -- random distribution in cubic domain
    // domainEdge constant specifies edge length
    for (int i = 0; i < nAtom; i++)
    {
        position[i].x = (T)(drand48() * domainEdge);
        position[i].y = (T)(drand48() * domainEdge);
        position[i].z = (T)(drand48() * domainEdge);
    }

    if (useTexture)
    {
        // Set up 1D texture to cache position info
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        // Bind a 1D texture to the position array
        CUDA_SAFE_CALL(cudaBindTexture(0, posTexture, d_position, channelDesc, nAtom*sizeof(float4)));

        cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<int4>();

        // Bind a 1D texture to the position array
        CUDA_SAFE_CALL(cudaBindTexture(0, posTexture_dp, d_position, channelDesc2, nAtom*sizeof(double4)));
    }

    // Keep track of how many atoms are within the cutoff distance to
    // accurately calculate FLOPS later
    int totalPairs = buildNeighborList<T, posVecType>(nAtom, position, neighborList);

    cout << "Finished.\n";
    cout << totalPairs << " of " << nAtom*maxNeighbors << " pairs within cutoff distance = " <<
            100.0 * ((double)totalPairs / (nAtom*maxNeighbors)) << " %" << endl;

    // Time the transfer of input data to the GPU
    cudaEvent_t inputTransfer_start, inputTransfer_stop;
    cudaEventCreate(&inputTransfer_start);
    cudaEventCreate(&inputTransfer_stop);

    cudaEventRecord(inputTransfer_start, 0);
    // Copy neighbor list data to GPU
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborList, neighborList, maxNeighbors*nAtom*sizeof(int), cudaMemcpyHostToDevice));
    // Copy position to GPU
    CUDA_SAFE_CALL(cudaMemcpy(d_position, position, nAtom*sizeof(posVecType), cudaMemcpyHostToDevice));
    cudaEventRecord(inputTransfer_stop, 0);
    CUDA_SAFE_CALL(cudaEventSynchronize(inputTransfer_stop));

    // Get elapsed time
    float inputTransfer_time = 0.0f;
    cudaEventElapsedTime(&inputTransfer_time, inputTransfer_start, inputTransfer_stop);
    inputTransfer_time *= 1.e-3;

    int blockSize = BLOCK_SIZE;
    int gridSize  = ceil((double)nAtom / (double)blockSize / (double) WORK_PER_THREAD);

    // Warm up the kernel and check correctness
    compute_lj_force
    #ifdef TEMPLATE
    <T, forceVecType, posVecType, useTexture, texReader>
    #endif
                    <<<gridSize, blockSize>>>
                    (d_force, d_position, maxNeighbors, d_neighborList, cutsq, lj1, lj2, nAtom);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Copy back forces
    cudaEvent_t outputTransfer_start, outputTransfer_stop;
    cudaEventCreate(&outputTransfer_start);
    cudaEventCreate(&outputTransfer_stop);

    cudaEventRecord(outputTransfer_start, 0);
    CUDA_SAFE_CALL(cudaMemcpy(force, d_force, nAtom*sizeof(forceVecType), cudaMemcpyDeviceToHost));
    cudaEventRecord(outputTransfer_stop, 0);
    CUDA_SAFE_CALL(cudaEventSynchronize(outputTransfer_stop));

    // Get elapsed time
    float outputTransfer_time = 0.0f;
    cudaEventElapsedTime(&outputTransfer_time, outputTransfer_start, outputTransfer_stop);
    outputTransfer_time *= 1.e-3;

    // If results are incorrect, skip the performance tests
    cout << "Performing Correctness Check (can take several minutes)\n";
    if (!checkResults<T, forceVecType, posVecType>(force, position, neighborList, nAtom))
    {
        return;
    }

    // Begin performance tests
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    int passes = op.getOptionInt("passes");
    int iter   = op.getOptionInt("iterations");

    for (int i = 0; i < passes; i++)
    {
        // Other kernels will be involved in true parallel versions
        cudaEventRecord(kernel_start, 0);
        for (int j = 0; j < iter; j++)
        {
            compute_lj_force
            #ifdef TEMPLATE
            <T, forceVecType, posVecType, useTexture, texReader>
            #endif
                <<<gridSize, blockSize>>>
                (d_force, d_position, maxNeighbors, d_neighborList, cutsq, lj1, lj2, nAtom);
        }
        cudaEventRecord(kernel_stop, 0);
        CUDA_SAFE_CALL(cudaEventSynchronize(kernel_stop));

        // get elapsed time
        float kernel_time = 0.0f;
        cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
        kernel_time /= (float)iter;
        kernel_time *= 1.e-3; // Convert to seconds
    }

    // Clean up
    // Host
    CUDA_SAFE_CALL(cudaFreeHost(position));
    CUDA_SAFE_CALL(cudaFreeHost(force));
    CUDA_SAFE_CALL(cudaFreeHost(neighborList));
    // Device
    CUDA_SAFE_CALL(cudaUnbindTexture(posTexture));
    CUDA_SAFE_CALL(cudaFree(d_position));
    CUDA_SAFE_CALL(cudaFree(d_force));
    CUDA_SAFE_CALL(cudaFree(d_neighborList));
    CUDA_SAFE_CALL(cudaEventDestroy(inputTransfer_start));
    CUDA_SAFE_CALL(cudaEventDestroy(inputTransfer_stop));
    CUDA_SAFE_CALL(cudaEventDestroy(outputTransfer_start));
    CUDA_SAFE_CALL(cudaEventDestroy(outputTransfer_stop));
    CUDA_SAFE_CALL(cudaEventDestroy(kernel_start));
    CUDA_SAFE_CALL(cudaEventDestroy(kernel_stop));
}
