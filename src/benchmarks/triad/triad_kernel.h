#ifndef TRIAD_KERNEL_H_
#define TRIAD_KERNEL_H_

#include<cuda.h>

// Select which precision that are used in the calculations
#if PRECISION == 32
    #define DATA_TYPE float
#elif PRECISION == 64
    #define DATA_TYPE double
#endif

__global__ void triad(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE s, int numberOfElements);

#endif
