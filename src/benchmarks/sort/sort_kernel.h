#ifndef SORT_KERNEL_H_
#define SORT_KERNEL_H_

#include <cuda.h>

#define WARP_SIZE 32

// Custom struct of two uint4s combined
typedef struct __builtin_align__(16) {
    uint4 a;
    uint4 b;
} uint8;

// Select data type based on data type size
// Scan data type size
#if SCAN_DATA_SIZE == 2
    #define SCAN_DATA_TYPE uint2
#elif SCAN_DATA_SIZE == 4
    #define SCAN_DATA_TYPE uint4
#elif SCAN_DATA_SIZE == 8
    #define SCAN_DATA_TYPE uint8
#endif

// Sort data type size
#if SORT_DATA_SIZE == 2
    #define SORT_DATA_TYPE uint2
#elif SORT_DATA_SIZE == 4
    #define SORT_DATA_TYPE uint4
#elif SORT_DATA_SIZE == 8
    #define SORT_DATA_TYPE uint8
#endif

typedef unsigned int uint;

__global__ void radixSortBlocks(uint nbits, uint startbit,
                                SORT_DATA_TYPE* keysOut, SORT_DATA_TYPE* valuesOut,
                                SORT_DATA_TYPE* keysIn, SORT_DATA_TYPE* valuesIn);

__global__ void findRadixOffsets(SCAN_DATA_TYPE* keys, uint* counters,
        uint* blockOffsets, uint startbit, uint numElements, uint totalBlocks);

__global__ void reorderData(uint startbit, uint *outKeys, uint *outValues,
        SCAN_DATA_TYPE *keys, SCAN_DATA_TYPE *values, uint *blockOffsets, uint *offsets,
        uint *sizes, uint totalBlocks);

// Scan Kernels
__global__ void vectorAddUniform4(uint *d_vector, const uint *d_uniforms, const int n);

__global__ void scan(uint *g_odata, uint *g_idata, uint *g_blockSums, const int n, const bool fullBlock, const bool storeSum);

#endif // SORT_KERNEL_H_
