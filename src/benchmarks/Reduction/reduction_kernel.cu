
extern "C"{

#if PRECISION == 32
#define T float
#else
#define T double
#endif

// The following class is a workaround for using dynamically sized
// shared memory in templated code. Without this workaround, the
// compiler would generate two shared memory arrays (one for SP
// and one for DP) of the same name and would generate an error.
class SharedMem
{
    public:
      __device__ inline T* getPointer()
      {
          extern __shared__ T s[];
          return s;
      };
};

// Reduction Kernel
__global__ void
reduce(const T* __restrict__ g_idata, T* __restrict__ g_odata, const unsigned int n)
{
    const unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x*(blockDim.x*2)) + tid;
    const unsigned int gridSize = blockDim.x*2*gridDim.x;

    // Shared memory will be used for intrablock summation
    // NB: CC's < 1.3 seem incompatible with the templated dynamic
    // shared memory workaround.
    // Inspection with cuda-gdb shows sdata as a pointer to *global*
    // memory and incorrect results are obtained.  This explicit macro
    // works around this issue. Further, it is acceptable to specify
    // float, since CC's < 1.3 do not support double precision.
// #if __CUDA_ARCH__ <= 130
    extern volatile __shared__ float sdata[];
// #else
    // SharedMem shared;
    // volatile T* sdata = shared.getPointer();
// #endif

    sdata[tid] = 0.0f;

    // Reduce multiple elements per thread
    while (i < n)
    {
        sdata[tid] += g_idata[i] + g_idata[i+BLOCK_SIZE];
        i += gridSize;
    }
    __syncthreads();

    // Reduce the contents of shared memory
    #if LOOP_UNROLL_REDUCE_1 == 1
    #pragma unroll 1
    #else
    #pragma unroll (LOOP_UNROLL_REDUCE_1)
    #endif
    for (int i = BLOCK_SIZE; i > 64; i /= 2) {
        if (tid < (i / 2)) {
            sdata[tid] += sdata[tid + (i / 2)];
        }
        __syncthreads();
    }

    if (tid < warpSize) {
        #if LOOP_UNROLL_REDUCE_2 == 1
        #pragma unroll 1
        #else
        #pragma unroll (LOOP_UNROLL_REDUCE_2)
        #endif
        for (int i = 64; i > 1; i /= 2) {
            if (BLOCK_SIZE >= i) {
                sdata[tid] += sdata[tid + (i / 2)];
                // NB2: This section would also need __sync calls if warp
                // synchronous execution were not assumed
            }
        }
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}


} // Extern C
