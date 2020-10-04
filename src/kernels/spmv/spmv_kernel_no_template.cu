#if PRECISION == 32
    #define fpType float
    #define texReader texReaderSP
#elif PRECISION == 64
    #define fpType double
    #define texReader texReaderDP
#endif

texture<float, 1> vecTex;  // vector textures
texture<int2, 1>  vecTexD;

// Texture Readers (used so kernels can be templated)
struct texReaderSP {
   __device__ __forceinline__ float operator()(const int idx) const
   {
       return tex1Dfetch(vecTex, idx);
   }
};

struct texReaderDP {
   __device__ __forceinline__ double operator()(const int idx) const
   {
       int2 v = tex1Dfetch(vecTexD, idx);
#if (__CUDA_ARCH__ < 130)
       // Devices before arch 130 don't support DP, and having the
       // __hiloint2double() intrinsic will cause compilation to fail.
       // This return statement added as a workaround -- it will compile,
       // but since the arch doesn't support DP, it will never be called
       return 0;
#else
       return __hiloint2double(v.y, v.x);
#endif
   }
};

// ****************************************************************************
// Function: spmv_csr_scalar_kernel
//
// Purpose:
//   Computes sparse matrix - vector multiplication on the GPU using
//   the CSR data storage format, using a thread per row of the sparse
//   matrix; based on Bell (SC09) and Baskaran (IBM Tech Report)
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element of the sparse matrix
//   rowDelimiters: array of size dim+1 holding indices to rows of the matrix
//                  last element is the index one past the last
//                  element of the matrix
//   dim: number of rows in the matrix
//   out: output - result from the spmv calculation
//
// Returns:  nothing
//           out indirectly through a pointer
//
// Programmer: Lukasz Wesolowski
// Creation: June 28, 2010
//
// Modifications:
//
// ****************************************************************************
extern "C" __global__ void
spmv_csr_scalar_kernel(const fpType * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       #if TEXTURE_MEMORY == 0
                       fpType * vec,
                       #endif
                       const int dim, fpType * __restrict__ out)
{
    int myRow = blockIdx.x * blockDim.x + threadIdx.x;
    #if TEXTURE_MEMORY == 1
    texReader vecTexReader;
    #endif

    if (myRow < dim)
    {
        fpType t = 0.0f;
        int start = rowDelimiters[myRow];
        int end = rowDelimiters[myRow+1];
        #if UNROLL_LOOP_1
        #pragma unroll
        #else
        #pragma unroll(1)
        #endif
        for (int j = start; j < end; j++)
        {
            int col = cols[j];
            #if TEXTURE_MEMORY == 0
            t += val[j] * vec[col];
            #else 
            t += val[j] * vecTexReader(col);
            #endif
        }
        out[myRow] = t;
    }
}

// ****************************************************************************
// Function: spmv_csr_vector_kernel
//
// Purpose:
//   Computes sparse matrix - vector multiplication on the GPU using
//   the CSR data storage format, using a warp per row of the sparse
//   matrix; based on Bell (SC09) and Baskaran (IBM Tech Report)
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element of the sparse matrix
//   rowDelimiters: array of size dim+1 holding indices to rows of the matrix
//                  last element is the index one past the last
//                  element of the matrix
//   dim: number of rows in the matrix
//   out: output - result from the spmv calculation
//
// Returns:  nothing
//           out indirectly through a pointer
//
// Programmer: Lukasz Wesolowski
// Creation: June 28, 2010
//
// Modifications:
//
// ****************************************************************************
extern "C" __global__ void
spmv_csr_vector_kernel(const fpType * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       #if TEXTURE_MEMORY == 0
                       fpType * vec,
                       #endif
                       const int dim, fpType * __restrict__ out)
{
    // Thread ID in block
    int t = threadIdx.x;
    // Thread ID within warp
    int id = t & (warpSize-1);
    int warpsPerBlock = blockDim.x / warpSize;
    // One row per warp
    int myRow = (blockIdx.x * warpsPerBlock) + (t / warpSize);
    // Texture reader for the dense vector
    #if TEXTURE_MEMORY == 1
    texReader vecTexReader;
    #endif

    __shared__ volatile fpType partialSums[BLOCK_SIZE];

    if (myRow < dim) {
        int warpStart = rowDelimiters[myRow];
        int warpEnd = rowDelimiters[myRow+1];
        fpType mySum = 0;
        #if UNROLL_LOOP_1
        #pragma unroll
        #else
        #pragma unroll(1)
        #endif
        for (int j = warpStart + id; j < warpEnd; j += warpSize)
        {
            int col = cols[j];
            #if TEXTURE_MEMORY == 0
            mySum += val[j] * vec[col]; 
            #else 
            mySum += val[j] * vecTexReader(col);
            #endif
        }
        partialSums[t] = mySum;

        // Reduce partial sums
        if (id < 16) {
            #if UNROLL_LOOP_2
            #pragma unroll
            #else
            #pragma unroll(1)
            #endif
            for (int i = 4; i >= 0; i--) {
                int l = 1<<i;
                if (id < l) partialSums[t] += partialSums[t+l];
            }
        }
        
        // Write result
        if (id == 0) {
            out[myRow] = partialSums[t];
        }
    }
}

// ****************************************************************************
// Function: spmv_ellpackr_kernel
//
// Purpose:
//   Computes sparse matrix - vector multiplication on the GPU using
//   the ELLPACK-R data storage format; based on Vazquez et al (Univ. of
//   Almeria Tech Report 2009)
//
// Arguments:
//   val: array holding the non-zero values for the matrix in column
//   major format and padded with zeros up to the length of longest row
//   cols: array of column indices for each element of the sparse matrix
//   rowLengths: array storing the length of each row of the sparse matrix
//   dim: number of rows in the matrix
//   out: output - result from the spmv calculation
//
// Returns:  nothing directly
//           out indirectly through a pointer
//
// Programmer: Lukasz Wesolowski
// Creation: June 29, 2010
//
// Modifications:
//
// ****************************************************************************
extern "C" __global__ void
spmv_ellpackr_kernel(const fpType * __restrict__ val,
                     const int    * __restrict__ cols,
                     const int    * __restrict__ rowLengths,
                     #if TEXTURE_MEMORY == 0
                     fpType * vec,
                     #endif
                     const int dim, fpType * __restrict__ out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    #if TEXTURE_MEMORY == 1
    texReader vecTexReader;
    #endif

    if (t < dim)
    {
        fpType result = 0.0f;
        int max = rowLengths[t];

        #if UNROLL_LOOP_1
        #pragma unroll
        #else
        #pragma unroll(1)
        #endif
        for (int i = 0; i < max; i++)
        {
            int ind = i*dim+t;
            #if TEXTURE_MEMORY == 0
            result += val[ind] * vec[cols[ind]];
            #else
            result += val[ind] * vecTexReader(cols[ind]);
            #endif
        }
        out[t] = result;
    }
}
