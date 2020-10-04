// This kernel code based on CUDPP.  Please see the notice in
// LICENSE_CUDPP.txt.

#include <cuda.h>
#include "sort_kernel.h"

__device__
#if INLINE_LSB
__forceinline__
#else
__noinline__ 
#endif
uint scanLSB(const uint val, uint* s_data)
{
    // Shared mem is "SORT_DATA_SIZE" * "SORT_BLOCK_SIZE" uints long, set first half to 0's
    int idx = threadIdx.x;
    s_data[idx] = 0;
    __syncthreads();

    // Set 2nd half to thread local sum (sum of the "SORT_DATA_SIZE" elements from global mem)
    idx += blockDim.x; // += "SORT_BLOCK_SIZE"

    // Some of these __sync's are unnecessary due to warp synchronous
    // execution.  Right now these are left in to be consistent with
    // opencl version, since that has to execute on platforms where
    // thread groups are not synchronous (i.e. CPUs)
    uint t;
    s_data[idx] = val;
    __syncthreads();

    #if LOOP_UNROLL_LSB
    #pragma unroll
    #else
    #pragma unroll(1)
    #endif
    for (uint i = 0; (SORT_BLOCK_SIZE >> i) > 1; i++) {
        t = s_data[idx - (1 << i)]; // (1 << i) = pow(2, i)
        __syncthreads();
        s_data[idx] += t;
        __syncthreads();
    }

    return s_data[idx] - val;  // convert inclusive -> exclusive
}

__device__
#if INLINE_SCAN
__forceinline__
#else
__noinline__ 
#endif
SORT_DATA_TYPE scan4(SORT_DATA_TYPE idata, uint* ptr)
{
    SORT_DATA_TYPE val4 = idata;
    SORT_DATA_TYPE sum;

    // Scan the "SORT_DATA_SIZE" elements in idata within this thread
#if SORT_DATA_SIZE == 2
    sum.x = val4.x;
    uint val = val4.y + sum.x;
#elif SORT_DATA_SIZE == 4
    sum.x = val4.x;
    sum.y = val4.y + sum.x;
    sum.z = val4.z + sum.y;
    uint val = val4.w + sum.z;
#elif SORT_DATA_SIZE == 8
    sum.a.x = val4.a.x;
    sum.a.y = val4.a.y + sum.a.x;
    sum.a.z = val4.a.z + sum.a.y;
    sum.a.w = val4.a.w + sum.a.z;
    sum.b.x = val4.b.x + sum.a.w;
    sum.b.y = val4.b.y + sum.b.x;
    sum.b.z = val4.b.z + sum.b.y;
    uint val = val4.b.w + sum.b.z;
#endif

    // Now scan those sums across the local work group
    val = scanLSB(val, ptr);

#if (SORT_DATA_SIZE == 2 || SORT_DATA_SIZE == 4)
    val4.x = val;
    val4.y = val + sum.x;
#endif
#if SORT_DATA_SIZE == 4
    val4.z = val + sum.y;
    val4.w = val + sum.z;
#elif SORT_DATA_SIZE == 8
    val4.a.x = val;
    val4.a.y = val + sum.a.x;
    val4.a.z = val + sum.a.y;
    val4.a.w = val + sum.a.z;
    val4.b.x = val + sum.a.w;
    val4.b.y = val + sum.b.x;
    val4.b.z = val + sum.b.y;
    val4.b.w = val + sum.b.z;
#endif

    return val4;
}

//----------------------------------------------------------------------------
//
// radixSortBlocks sorts all blocks of data independently in shared
// memory.  Each thread block (CTA) sorts one block of "SORT_DATA_SIZE"*CTA_SIZE elements
//
// The radix sort is done in two stages.  This stage calls radixSortBlock
// on each block independently, sorting on the basis of bits
// (startbit) -> (startbit + nbits)
//----------------------------------------------------------------------------

__global__ void radixSortBlocks(const uint nbits, const uint startbit,
                                SORT_DATA_TYPE* keysOut, SORT_DATA_TYPE* valuesOut,
                                SORT_DATA_TYPE* keysIn, SORT_DATA_TYPE* valuesIn)
{
    __shared__ uint sMem[SORT_DATA_SIZE * SORT_BLOCK_SIZE];

    // Get Indexing information
    const uint i = threadIdx.x + (blockIdx.x * blockDim.x);
    const uint tid = threadIdx.x;
    const uint localSize = blockDim.x;

    // Load keys and vals from global memory
    SORT_DATA_TYPE key, value;
    key = keysIn[i];
    value = valuesIn[i];

    // For each of the 4 bits
    for (uint shift = startbit; shift < (startbit + nbits); ++shift)
    {
        // Check if the LSB is 0
        SORT_DATA_TYPE lsb;
    #if (SORT_DATA_SIZE == 2 || SORT_DATA_SIZE == 4)
        lsb.x = !((key.x >> shift) & 0x1);
        lsb.y = !((key.y >> shift) & 0x1);
    #endif
    #if SORT_DATA_SIZE == 4
        lsb.z = !((key.z >> shift) & 0x1);
        lsb.w = !((key.w >> shift) & 0x1);
    #elif SORT_DATA_SIZE == 8
        lsb.a.x = !((key.a.x >> shift) & 0x1);
        lsb.a.y = !((key.a.y >> shift) & 0x1);
        lsb.a.z = !((key.a.z >> shift) & 0x1);
        lsb.a.w = !((key.a.w >> shift) & 0x1);
        lsb.b.x = !((key.b.x >> shift) & 0x1);
        lsb.b.y = !((key.b.y >> shift) & 0x1);
        lsb.b.z = !((key.b.z >> shift) & 0x1);
        lsb.b.w = !((key.b.w >> shift) & 0x1);
    #endif

        // Do an exclusive scan of how many elems have 0's in the LSB
        // When this is finished, address.n will contain the number of
        // elems with 0 in the LSB which precede elem n
        SORT_DATA_TYPE address = scan4(lsb, sMem);

        __shared__ uint numtrue;

        // Store the total number of elems with an LSB of 0
        // to shared mem
        if (tid == localSize - 1)
        {
        #if SORT_DATA_SIZE == 2
            numtrue = address.y + lsb.y;
        #elif SORT_DATA_SIZE == 4
            numtrue = address.w + lsb.w;
        #elif SORT_DATA_SIZE == 8
            numtrue = address.b.w + lsb.b.w;
        #endif
        }
        __syncthreads();

        // Determine rank -- position in the block
        // If you are a 0 --> your position is the scan of 0's
        // If you are a 1 --> your position is calculated as below
        SORT_DATA_TYPE rank;
        const int idx = tid * SORT_DATA_SIZE;

    #if (SORT_DATA_SIZE == 2 || SORT_DATA_SIZE == 4)
        rank.x = lsb.x ? address.x : numtrue + idx     - address.x;
        rank.y = lsb.y ? address.y : numtrue + idx + 1 - address.y;
    #endif
    #if SORT_DATA_SIZE == 4
        rank.z = lsb.z ? address.z : numtrue + idx + 2 - address.z;
        rank.w = lsb.w ? address.w : numtrue + idx + 3 - address.w;
    #elif SORT_DATA_SIZE == 8
        rank.a.x = lsb.a.x ? address.a.x : numtrue + idx     - address.a.x;
        rank.a.y = lsb.a.y ? address.a.y : numtrue + idx + 1 - address.a.y;
        rank.a.z = lsb.a.z ? address.a.z : numtrue + idx + 2 - address.a.z;
        rank.a.w = lsb.a.w ? address.a.w : numtrue + idx + 3 - address.a.w;
        rank.b.x = lsb.b.x ? address.b.x : numtrue + idx + 4 - address.b.x;
        rank.b.y = lsb.b.y ? address.b.y : numtrue + idx + 5 - address.b.y;
        rank.b.z = lsb.b.z ? address.b.z : numtrue + idx + 6 - address.b.z;
        rank.b.w = lsb.b.w ? address.b.w : numtrue + idx + 7 - address.b.w;
    #endif

        // Scatter keys into local mem
    #if SORT_DATA_SIZE == 2
        sMem[(rank.x & 1) * localSize + (rank.x >> 1)] = key.x;
        sMem[(rank.y & 1) * localSize + (rank.y >> 1)] = key.y;
    #elif SORT_DATA_SIZE == 4
        sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = key.x;
        sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = key.y;
        sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = key.z;
        sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = key.w;
    #elif SORT_DATA_SIZE == 8
        sMem[(rank.a.x & 7) * localSize + (rank.a.x >> 3)] = key.a.x;
        sMem[(rank.a.y & 7) * localSize + (rank.a.y >> 3)] = key.a.y;
        sMem[(rank.a.z & 7) * localSize + (rank.a.z >> 3)] = key.a.z;
        sMem[(rank.a.w & 7) * localSize + (rank.a.w >> 3)] = key.a.w;
        sMem[(rank.b.x & 7) * localSize + (rank.b.x >> 3)] = key.b.x;
        sMem[(rank.b.y & 7) * localSize + (rank.b.y >> 3)] = key.b.y;
        sMem[(rank.b.z & 7) * localSize + (rank.b.z >> 3)] = key.b.z;
        sMem[(rank.b.w & 7) * localSize + (rank.b.w >> 3)] = key.b.w;
    #endif

        __syncthreads();

        // Read keys out of local mem into registers, in prep for
        // write out to global mem
    #if (SORT_DATA_SIZE == 2 || SORT_DATA_SIZE == 4)
        key.x = sMem[tid];
        key.y = sMem[tid +     localSize];
    #endif
    #if SORT_DATA_SIZE == 4
        key.z = sMem[tid + 2 * localSize];
        key.w = sMem[tid + 3 * localSize];
    #elif SORT_DATA_SIZE == 8
        key.a.x = sMem[tid];
        key.a.y = sMem[tid +     localSize];
        key.a.z = sMem[tid + 2 * localSize];
        key.a.w = sMem[tid + 3 * localSize];
        key.b.x = sMem[tid + 4 * localSize];
        key.b.y = sMem[tid + 5 * localSize];
        key.b.z = sMem[tid + 6 * localSize];
        key.b.w = sMem[tid + 7 * localSize];
    #endif

        __syncthreads();

        // Scatter values into local mem
    #if SORT_DATA_SIZE == 2
        sMem[(rank.x & 1) * localSize + (rank.x >> 1)] = value.x;
        sMem[(rank.y & 1) * localSize + (rank.y >> 1)] = value.y;
    #elif SORT_DATA_SIZE == 4
        sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = value.x;
        sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = value.y;
        sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = value.z;
        sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = value.w;
    #elif SORT_DATA_SIZE == 8
        sMem[(rank.a.x & 7) * localSize + (rank.a.x >> 3)] = value.a.x;
        sMem[(rank.a.y & 7) * localSize + (rank.a.y >> 3)] = value.a.y;
        sMem[(rank.a.z & 7) * localSize + (rank.a.z >> 3)] = value.a.z;
        sMem[(rank.a.w & 7) * localSize + (rank.a.w >> 3)] = value.a.w;
        sMem[(rank.b.x & 7) * localSize + (rank.b.x >> 3)] = value.b.x;
        sMem[(rank.b.y & 7) * localSize + (rank.b.y >> 3)] = value.b.y;
        sMem[(rank.b.z & 7) * localSize + (rank.b.z >> 3)] = value.b.z;
        sMem[(rank.b.w & 7) * localSize + (rank.b.w >> 3)] = value.b.w;
    #endif

        __syncthreads();

        // Read keys out of local mem into registers, in prep for
        // write out to global mem
    #if (SORT_DATA_SIZE == 2 || SORT_DATA_SIZE == 4)
        value.x = sMem[tid];
        value.y = sMem[tid +     localSize];
    #endif
    #if SORT_DATA_SIZE == 4
        value.z = sMem[tid + 2 * localSize];
        value.w = sMem[tid + 3 * localSize];
    #elif SORT_DATA_SIZE == 8
        value.a.x = sMem[tid];
        value.a.y = sMem[tid +     localSize];
        value.a.z = sMem[tid + 2 * localSize];
        value.a.w = sMem[tid + 3 * localSize];
        value.b.x = sMem[tid + 4 * localSize];
        value.b.y = sMem[tid + 5 * localSize];
        value.b.z = sMem[tid + 6 * localSize];
        value.b.w = sMem[tid + 7 * localSize];
    #endif

        __syncthreads();
    }
    keysOut[i]   = key;
    valuesOut[i] = value;
}

//----------------------------------------------------------------------------
// Given an array with blocks sorted according to a 4-bit radix group, each
// block counts the number of keys that fall into each radix in the group, and
// finds the starting offset of each radix in the block.  It then writes the
// radix counts to the counters array, and the starting offsets to the
// blockOffsets array.
//
//----------------------------------------------------------------------------
__global__ void findRadixOffsets(SCAN_DATA_TYPE* keys, uint* counters,
        uint* blockOffsets, uint startbit, uint numElements, uint totalBlocks)
{
    __shared__ uint  sStartPointers[16];
    extern __shared__ uint sRadix1[];

    uint groupId = blockIdx.x;
    uint localId = threadIdx.x;
    uint groupSize = blockDim.x;

    SCAN_DATA_TYPE radix2;
    radix2 = keys[threadIdx.x + (blockIdx.x * blockDim.x)];

#if SCAN_DATA_SIZE == 2
    sRadix1[2 * localId]     = (radix2.x >> startbit) & 0xF;
    sRadix1[2 * localId + 1] = (radix2.y >> startbit) & 0xF;
#elif SCAN_DATA_SIZE == 4
    sRadix1[4 * localId]     = (radix2.x >> startbit) & 0xF;
    sRadix1[4 * localId + 1] = (radix2.y >> startbit) & 0xF;
    sRadix1[4 * localId + 2] = (radix2.z >> startbit) & 0xF;
    sRadix1[4 * localId + 3] = (radix2.w >> startbit) & 0xF;
#elif SCAN_DATA_SIZE == 8
    sRadix1[8 * localId]     = (radix2.a.x >> startbit) & 0xF;
    sRadix1[8 * localId + 1] = (radix2.a.y >> startbit) & 0xF;
    sRadix1[8 * localId + 2] = (radix2.a.z >> startbit) & 0xF;
    sRadix1[8 * localId + 3] = (radix2.a.w >> startbit) & 0xF;
    sRadix1[8 * localId + 4] = (radix2.b.x >> startbit) & 0xF;
    sRadix1[8 * localId + 5] = (radix2.b.y >> startbit) & 0xF;
    sRadix1[8 * localId + 6] = (radix2.b.z >> startbit) & 0xF;
    sRadix1[8 * localId + 7] = (radix2.b.w >> startbit) & 0xF;
#endif

    // Finds the position where the sRadix1 entries differ and stores start
    // index for each radix.
    if(localId < 16)
    {
        sStartPointers[localId] = 0;
    }
    __syncthreads();

    if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
    {
        sStartPointers[sRadix1[localId]] = localId;
    }

    if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1])
    {
        sStartPointers[sRadix1[localId + groupSize]] = localId + groupSize;
    }

#if SCAN_DATA_SIZE >= 4
    for (uint i = 2; i < SCAN_DATA_SIZE; i++) {
        if(sRadix1[localId + i * groupSize] != sRadix1[localId + i * groupSize - 1])
        {
            sStartPointers[sRadix1[localId + i * groupSize]] = localId + i * groupSize;
        }
    }
#endif

    __syncthreads();

    if(localId < 16)
    {
        blockOffsets[groupId*16 + localId] = sStartPointers[localId];
    }
    __syncthreads();

    // Compute the sizes of each block.
    if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
    {
        sStartPointers[sRadix1[localId - 1]] = localId - sStartPointers[sRadix1[localId - 1]];
    }

    if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1] )
    {
        sStartPointers[sRadix1[localId + groupSize - 1]] = localId + groupSize - sStartPointers[sRadix1[localId + groupSize - 1]];
    }

#if SCAN_DATA_SIZE >= 4
    for (uint i = 2; i < SCAN_DATA_SIZE; i++) {
        if(sRadix1[localId + i * groupSize] != sRadix1[localId + i * groupSize - 1] )
        {
            sStartPointers[sRadix1[localId + i * groupSize - 1]] = localId + i * groupSize - sStartPointers[sRadix1[localId + i * groupSize - 1]];
        }
    }
#endif

    if(localId == groupSize - 1)
    {
        sStartPointers[sRadix1[SCAN_DATA_SIZE * groupSize - 1]] = SCAN_DATA_SIZE * groupSize - sStartPointers[sRadix1[SCAN_DATA_SIZE * groupSize - 1]];
    }
    __syncthreads();

    if(localId < 16)
    {
        counters[localId * totalBlocks + groupId] = sStartPointers[localId];
    }
}

//----------------------------------------------------------------------------
// reorderData shuffles data in the array globally after the radix offsets
// have been found. On compute version 1.1 and earlier GPUs, this code depends
// on SORT_BLOCK_SIZE being 16 * number of radices (i.e. 16 * 2^nbits).
//----------------------------------------------------------------------------
__global__ void reorderData(uint  startbit,
                            uint  *outKeys,
                            uint  *outValues,
                            SCAN_DATA_TYPE *keys,
                            SCAN_DATA_TYPE *values,
                            uint  *blockOffsets,
                            uint  *offsets,
                            uint  *sizes,
                            uint  totalBlocks)
{
    uint GROUP_SIZE = blockDim.x;
    __shared__ SCAN_DATA_TYPE sKeys2[SCAN_BLOCK_SIZE];
    __shared__ SCAN_DATA_TYPE sValues2[SCAN_BLOCK_SIZE];
    __shared__ uint  sOffsets[16];
    __shared__ uint  sBlockOffsets[16];
    uint* sKeys1   = (uint*) sKeys2;
    uint* sValues1 = (uint*) sValues2;

    uint blockId = blockIdx.x;

    uint i = blockId * blockDim.x + threadIdx.x;

    sKeys2[threadIdx.x]   = keys[i];
    sValues2[threadIdx.x] = values[i];

    if(threadIdx.x < 16)
    {
        sOffsets[threadIdx.x]      = offsets[threadIdx.x * totalBlocks + blockId];
        sBlockOffsets[threadIdx.x] = blockOffsets[blockId * 16 + threadIdx.x];
    }
    __syncthreads();

    uint radix = (sKeys1[threadIdx.x] >> startbit) & 0xF;
    uint globalOffset = sOffsets[radix] + threadIdx.x - sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[threadIdx.x];
    outValues[globalOffset] = sValues1[threadIdx.x];

    radix = (sKeys1[threadIdx.x + GROUP_SIZE] >> startbit) & 0xF;
    globalOffset = sOffsets[radix] + threadIdx.x + GROUP_SIZE - sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[threadIdx.x + GROUP_SIZE];
    outValues[globalOffset] = sValues1[threadIdx.x + GROUP_SIZE];

#if SCAN_DATA_SIZE >= 4
    for (uint i = 2; i < SCAN_DATA_SIZE; i++) {
        radix = (sKeys1[threadIdx.x + i * GROUP_SIZE] >> startbit) & 0xF;
        globalOffset = sOffsets[radix] + threadIdx.x + i * GROUP_SIZE - sBlockOffsets[radix];

        outKeys[globalOffset]   = sKeys1[threadIdx.x + i * GROUP_SIZE];
        outValues[globalOffset] = sValues1[threadIdx.x + i * GROUP_SIZE];
    }
#endif
}

__device__
#if INLINE_LOCAL_MEMORY
__forceinline__
#else
__noinline__ 
#endif
uint scanLocalMem(const uint val, uint* s_data)
{
    // Shared mem is 512 uints long, set first half to 0
    int idx = threadIdx.x;
    s_data[idx] = 0.0f;
    __syncthreads();

    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)
    idx += blockDim.x; // += "SCAN_BLOCK_SIZE"

    // Some of these __sync's are unnecessary due to warp synchronous
    // execution.  Right now these are left in to be consistent with
    // opencl version, since that has to execute on platforms where
    // thread groups are not synchronous (i.e. CPUs)
    uint t;
    s_data[idx] = val;
    __syncthreads();
    
    #if LOOP_UNROLL_LOCAL_MEMORY
    #pragma unroll
    #else
    #pragma unroll(1)
    #endif
    for (uint i = 0; (SCAN_BLOCK_SIZE >> i) > 1; i++) {
        t = s_data[idx - (1 << i)]; // (1 << i) = pow(2, i)
        __syncthreads();
        s_data[idx] += t;
        __syncthreads();
    }

    return s_data[idx-1];
}

__global__ void
scan(uint *g_odata, uint* g_idata, uint* g_blockSums, const int n, const bool fullBlock, const bool storeSum)
{
    __shared__ uint s_data[SCAN_DATA_SIZE * SCAN_BLOCK_SIZE];

    // Load data into shared mem
    SORT_DATA_TYPE tempData;
    SORT_DATA_TYPE threadScanT;
    uint res;
    SORT_DATA_TYPE* inData  = (SORT_DATA_TYPE*) g_idata;

    const int gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int tid = threadIdx.x;
    const int i = gid * SORT_DATA_SIZE;

    // If possible, read from global mem in a "SORT_DATA_TYPE" chunk
#if SORT_DATA_SIZE == 2
    if (fullBlock || i + 1 < n)
    {
        // scan the 2 elements read in from global
        tempData       = inData[gid];
        threadScanT.x = tempData.x;
        threadScanT.y = tempData.y + threadScanT.x;
        res = threadScanT.y;
    }
    else
    {   // if not, read individual uints, scan & store in lmem
        threadScanT.x = (i < n) ? g_idata[i] : 0.0f;
        threadScanT.y = ((i+1 < n) ? g_idata[i+1] : 0.0f) + threadScanT.x;
        res = threadScanT.y;
    }
#elif SORT_DATA_SIZE == 4
    if (fullBlock || i + 3 < n)
    {
        // scan the 4 elements read in from global
        tempData       = inData[gid];
        threadScanT.x = tempData.x;
        threadScanT.y = tempData.y + threadScanT.x;
        threadScanT.z = tempData.z + threadScanT.y;
        threadScanT.w = tempData.w + threadScanT.z;
        res = threadScanT.w;
    }
    else
    {   // if not, read individual uints, scan & store in lmem
        threadScanT.x = (i < n) ? g_idata[i] : 0.0f;
        threadScanT.y = ((i+1 < n) ? g_idata[i+1] : 0.0f) + threadScanT.x;
        threadScanT.z = ((i+2 < n) ? g_idata[i+2] : 0.0f) + threadScanT.y;
        threadScanT.w = ((i+3 < n) ? g_idata[i+3] : 0.0f) + threadScanT.z;
        res = threadScanT.w;
    }
#elif SORT_DATA_SIZE == 8
    if (fullBlock || i + 7 < n)
    {
        // scan the 8 elements read in from global
        tempData       = inData[gid];
        threadScanT.a.x = tempData.a.x;
        threadScanT.a.y = tempData.a.y + threadScanT.a.x;
        threadScanT.a.z = tempData.a.z + threadScanT.a.y;
        threadScanT.a.w = tempData.a.w + threadScanT.a.z;
        threadScanT.b.x = tempData.b.x + threadScanT.a.w;
        threadScanT.b.y = tempData.b.y + threadScanT.b.x;
        threadScanT.b.z = tempData.b.z + threadScanT.b.y;
        threadScanT.b.w = tempData.b.w + threadScanT.b.z;
        res = threadScanT.b.w;
    }
    else
    {   // if not, read individual uints, scan & store in lmem
        threadScanT.a.x = (i < n) ? g_idata[i] : 0.0f;
        threadScanT.a.y = ((i+1 < n) ? g_idata[i+1] : 0.0f) + threadScanT.a.x;
        threadScanT.a.z = ((i+2 < n) ? g_idata[i+2] : 0.0f) + threadScanT.a.y;
        threadScanT.a.w = ((i+3 < n) ? g_idata[i+3] : 0.0f) + threadScanT.a.z;
        threadScanT.b.x = ((i+4 < n) ? g_idata[i+4] : 0.0f) + threadScanT.a.w;
        threadScanT.b.y = ((i+5 < n) ? g_idata[i+5] : 0.0f) + threadScanT.b.x;
        threadScanT.b.z = ((i+6 < n) ? g_idata[i+6] : 0.0f) + threadScanT.b.y;
        threadScanT.b.w = ((i+7 < n) ? g_idata[i+7] : 0.0f) + threadScanT.b.z;
        res = threadScanT.b.w;
    }
#endif

    res = scanLocalMem(res, s_data);
    __syncthreads();

    // If we have to store the sum for the block, have the last work item
    // in the block write it out
    if (storeSum && tid == blockDim.x-1) {
    #if SORT_DATA_SIZE == 2
        g_blockSums[blockIdx.x] = res + threadScanT.y;
    #elif SORT_DATA_SIZE == 4
        g_blockSums[blockIdx.x] = res + threadScanT.w;
    #elif SORT_DATA_SIZE == 8
        g_blockSums[blockIdx.x] = res + threadScanT.b.w;
    #endif
    }

    // write results to global memory
    SORT_DATA_TYPE* outData = (SORT_DATA_TYPE*) g_odata;

#if SORT_DATA_SIZE == 2
    tempData.x = res;
    tempData.y = res + threadScanT.x;

    if (fullBlock || i + 1 < n)
    {
        outData[gid] = tempData;
    }
    else
    {
        if ( i    < n) { g_odata[i]   = tempData.x; }
    }
#elif SORT_DATA_SIZE == 4
    tempData.x = res;
    tempData.y = res + threadScanT.x;
    tempData.z = res + threadScanT.y;
    tempData.w = res + threadScanT.z;

    if (fullBlock || i + 3 < n)
    {
        outData[gid] = tempData;
    }
    else
    {
        if ( i    < n) { g_odata[i]   = tempData.x;
        if ((i+1) < n) { g_odata[i+1] = tempData.y;
        if ((i+2) < n) { g_odata[i+2] = tempData.z; } } }
    }
#elif SORT_DATA_SIZE == 8
    tempData.a.x = res;
    tempData.a.y = res + threadScanT.a.x;
    tempData.a.z = res + threadScanT.a.y;
    tempData.a.w = res + threadScanT.a.z;
    tempData.b.x = res + threadScanT.a.w;
    tempData.b.y = res + threadScanT.b.x;
    tempData.b.z = res + threadScanT.b.y;
    tempData.b.w = res + threadScanT.b.z;

    if (fullBlock || i + 7 < n)
    {
        outData[gid] = tempData;
    }
    else
    {
        if ( i    < n) { g_odata[i]   = tempData.a.x;
        if ((i+1) < n) { g_odata[i+1] = tempData.a.y;
        if ((i+2) < n) { g_odata[i+2] = tempData.a.z;
        if ((i+3) < n) { g_odata[i+3] = tempData.a.w;
        if ((i+4) < n) { g_odata[i+4] = tempData.b.x;
        if ((i+5) < n) { g_odata[i+5] = tempData.b.y;
        if ((i+6) < n) { g_odata[i+6] = tempData.b.z; } } } } } } }
    }
#endif
}

__global__ void
vectorAddUniform4(uint *d_vector, const uint *d_uniforms, const int n)
{
    __shared__ uint uni[1];

    if (threadIdx.x == 0)
    {
        uni[0] = d_uniforms[blockIdx.x];
    }

    unsigned int address = threadIdx.x + (blockIdx.x * blockDim.x * SORT_DATA_SIZE);

    __syncthreads();

    // "SORT_DATA_SIZE" elements per thread
    #if LOOP_UNROLL_ADD_UNIFORM
    #pragma unroll
    #else
    #pragma unroll(1)
    #endif
    for (int i = 0; i < SORT_DATA_SIZE && address < n; i++)
    {
        d_vector[address] += uni[0];
        address += blockDim.x;
    }
}
