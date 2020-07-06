#if PRECISION == 32 
    // Texture Readers
    texture<float, 1> vecTex;    

    struct texReader {
        __device__ __forceinline__ float operator()(const int idx) const
        {
            return tex1D(vecTex, idx);
        }
    };
#elif PRECISION == 64
    // Texture Readers
    texture<float, 1> vecTex;
    struct texReader {
        __device__ __forceinline__ double operator()(const int idx) const
        {
            return tex1D(vecTex, idx);
        }
     };
#endif


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
// template <typename fpType, typename texReader>
extern "C" __global__ void
spmv_csr_scalar_kernel( float * valSP,
                         double * valDP,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, float * outSP,
                       double * outDP)
{
    int myRow = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    texReader vecTexReader;

    if (myRow < dim)
    {
        #if PRECISION == 32
            float t = 0.0f;
        #else 
            double t = 0.0;
        #endif
        int start = rowDelimiters[myRow];
        int end = rowDelimiters[myRow+1];
        #if UNROLL_LOOP
        #pragma unroll
        #endif

        for (int j = start; j < end; j++)
        {
            int col = cols[j];

            #if PRECISION == 32
                t += valSP[j] * vecTexReader(col);
            #else 
                t += valDP[j]* vecTexReader(col);
            #endif
        }
        #if PRECISION == 32
            outSP[myRow] = t;
        #else 
            outDP[myRow] = t;
        #endif
        
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
template <typename fpType, typename texReader>
__global__ void
spmv_csr_vector_kernel(const fpType * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
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
    texReader vecTexReader;

    __shared__ volatile fpType partialSums[BLOCK_SIZE];

    if (myRow < dim) {
        int warpStart = rowDelimiters[myRow];
        int warpEnd = rowDelimiters[myRow+1];
        fpType mySum = 0;
        #if UNROLL_LOOP
        #pragma unroll
        #endif
        for (int j = warpStart + id; j < warpEnd; j += warpSize)
        {
            int col = cols[j];
            mySum += val[j] * vecTexReader(col);
        }
        partialSums[t] = mySum;

        // Reduce partial sums
        if (id < 16) {
            #if UNROLL_LOOP_2
            #pragma unroll
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
template <typename fpType, typename texReader>
__global__ void
spmv_ellpackr_kernel(const fpType * __restrict__ val,
                     const int    * __restrict__ cols,
                     const int    * __restrict__ rowLengths,
                     const int dim, fpType * __restrict__ out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    texReader vecTexReader;

    if (t < dim)
    {
        fpType result = 0.0f;
        int max = rowLengths[t];

        #if UNROLL_LOOP
        #pragma unroll
        #endif
        for (int i = 0; i < max; i++)
        {
            int ind = i*dim+t;
            result += val[ind] * vecTexReader(cols[ind]);
        }
        out[t] = result;
    }
}

