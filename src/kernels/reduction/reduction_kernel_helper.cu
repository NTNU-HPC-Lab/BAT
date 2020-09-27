typedef unsigned int uint;

// Select which precision that are used in the calculations
// And define the replacements for the template inputs
#if PRECISION == 32
    #define T float
#elif PRECISION == 64
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

__inline__ __device__ double convertTextureObjectToDouble(cudaTextureObject_t textureObject, const int &position) {
    uint2 values = tex1Dfetch<uint2>(textureObject, position);
    return __hiloint2double(values.y, values.x);
}

// Reduction Kernel
extern "C"
__global__ void
reduce(const T* __restrict__ g_idata, cudaTextureObject_t idataTextureObject, T* __restrict__ g_odata, const unsigned int n)
{
    int blockSize = BLOCK_SIZE;

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
    
    // Fallback "pseudo-parameter" to set shared memory size in the kernel rather than in the host code
    // This is not an actual parameter, but rather a way of specifying that the kernel should set the size of the shared memory
    // It sets the size equal to the block size
    // This is needed because there is no way to set the shared memory from CLTune
#if KERNEL_SHARED_MEMORY_SIZE
    extern volatile __shared__ float sdata[BLOCK_SIZE];
#else
#if __CUDA_ARCH__ <= 130
    extern volatile __shared__ float sdata[];
#else
    SharedMem shared;
    volatile T* sdata = shared.getPointer();
#endif
#endif

    sdata[tid] = 0.0f;

    // Reduce multiple elements per thread
    while (i < n)
    {
    #if TEXTURE_MEMORY
        #if PRECISION == 32
            sdata[tid] += tex1Dfetch<T>(idataTextureObject, i) + tex1Dfetch<T>(idataTextureObject, i+blockSize);
        #elif PRECISION == 64
            sdata[tid] += convertTextureObjectToDouble(idataTextureObject, i) + convertTextureObjectToDouble(idataTextureObject, i+blockSize);
        #endif
    #else
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
    #endif
        i += gridSize;
    }
    __syncthreads();

    // Reduce the contents of shared memory
    #if LOOP_UNROLL_REDUCE_1
    #pragma unroll
    #else
    #pragma unroll(1)
    #endif
    for (int i = BLOCK_SIZE; i > 64; i /= 2) {
        if (blockSize >= i) {
            if (tid < (i / 2)) {
                sdata[tid] += sdata[tid + (i / 2)];
            }
            __syncthreads();
        }
    }

    #if LOOP_UNROLL_REDUCE_2
    #pragma unroll
    #else
    #pragma unroll(1)
    #endif
    for (int i = 64; i > 1; i /= 2) {
        if (tid < warpSize) {
            if (blockSize >= i) {
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

extern "C" __global__ void reduce_helper(
    const float* __restrict__ g_idataf, cudaTextureObject_t idataTextureObjectf, float* __restrict__ g_odataf,
    const double* __restrict__ g_idatad, cudaTextureObject_t idataTextureObjectd, double* __restrict__ g_odatad,
    const unsigned int n
) {
    #if PRECISION == 32
        reduce(g_idataf, idataTextureObjectf, g_odataf, n);
    #elif PRECISION == 64
        reduce(g_idatad, idataTextureObjectd, g_odatad, n);
    #endif
}
