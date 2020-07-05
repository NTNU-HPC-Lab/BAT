#ifndef TRIAD_KERNEL_H_
#define TRIAD_KERNEL_H_

#include<cuda.h>

__global__ void triad(float* A, float* B, float* C, float s);

#endif
