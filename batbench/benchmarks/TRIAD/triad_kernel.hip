#include "hip/hip_runtime.h"

// Select which precision that are used in the calculations
#ifndef PRECSION
#define PRECISION 32
#endif

#if PRECISION == 32
    #define DATA_TYPE float
#elif PRECISION == 64
    #define DATA_TYPE double
#endif

// ****************************************************************************
// Function: triad
//
// Purpose:
//   A simple vector addition kernel
//   C = A + s*B
//
// Arguments:
//   A,B - input vectors
//   C - output vectors
//   s - scalar
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
extern "C" __global__ void triad(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE s, int numberOfElements)
{
    int gid = (threadIdx.x + (blockIdx.x * blockDim.x)) * WORK_PER_THREAD;

    #if LOOP_UNROLL_TRIAD == 1
    #pragma unroll 1
    #else
    #pragma unroll(LOOP_UNROLL_TRIAD)
    #endif
    for (int i = 0; i < WORK_PER_THREAD; i++) {
        int threadId = gid + i;

        // Ensure that the current thread id is less than total number of elements
        if (threadId < numberOfElements) {
            C[threadId] = A[threadId] + s*B[threadId];
        }
    }
}
