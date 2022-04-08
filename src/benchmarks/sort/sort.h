#ifndef SORT_H_
#define SORT_H_

#include "sort_kernel.h"

typedef unsigned int uint;

static const int SORT_BITS = 32;

void
radixSortStep(uint nbits, uint startbit, SORT_DATA_TYPE* keys, SORT_DATA_TYPE* values,
        SORT_DATA_TYPE* tempKeys, SORT_DATA_TYPE* tempValues, uint* counters,
        uint* countersSum, uint* blockOffsets, uint** scanBlockSums,
        uint numElements);

void
scanArrayRecursive(uint* outArray, uint* inArray, int numElements, int level, uint** blockSums);

bool
verifySort(uint *keys, uint* vals, const size_t size);

#ifdef __DEVICE_EMULATION__
#define __SYNC __syncthreads();
#else
#define __SYNC ;
#endif

#endif // SORT_H_
