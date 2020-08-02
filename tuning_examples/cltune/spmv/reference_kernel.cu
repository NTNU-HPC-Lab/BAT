template <typename fpType, typename texReader>
__device__ void
spmv_csr_scalar_kernel(const fpType * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       fpType * vec,
                       const int dim, fpType * __restrict__ out);

template <typename fpType, typename texReader>
__device__ void
spmv_csr_vector_kernel(const fpType * __restrict__ val,
                        const int    * __restrict__ cols,
                        const int    * __restrict__ rowDelimiters,
                        fpType * vec,
                        const int dim, fpType * __restrict__ out);

template <typename fpType, typename texReader>
__device__ void
spmv_ellpackr_kernel(const fpType * __restrict__ val,
                        const int    * __restrict__ cols,
                        const int    * __restrict__ rowLengths,
                        fpType * vec,
                        const int dim, fpType * __restrict__ out);

// Texture Readers
texture<float, 1> vecTex;    

struct texReader {
    __device__ __forceinline__ float operator()(const int idx) const
    {
        return tex1D(vecTex, idx);
    }
};


/**
 * Helper function for tuners that can not use templated kernels directly
 * This function also chooses format based on a parameter
 */
extern "C" __global__ void
spmv_kernel(float * valSP_csr,
            double * valDP_csr,
            float * valSP_csr_pad,
            double * valDP_csr_pad,
            float * valSP_ellpackr,
            double * valDP_ellpackr,            
            const int    * __restrict__ cols_csr,
            const int    * __restrict__ cols_csr_pad,
            const int    * __restrict__ cols_ellpackr,
            const int    * __restrict__ rowDelimiters,
            const int    * __restrict__ rowDelimiters_pad,
            const int    * __restrict__ rowLengths,
            float * vecSP,
            double * vecDP,
            const int dim, 
            const int dim_pad, 
            float * outSP,
            double * outDP
        ) {

    #if PRECISION == 32
    // Single precision
        #if (FORMAT == 1 || FORMAT == 2) 
        // CSR scalar
            #if (FORMAT == 1) 
            // Normal
                spmv_csr_scalar_kernel<float, texReader>(valSP_csr, cols_csr, rowDelimiters, vecSP, dim, outSP);
            #else 
            // Padded
                spmv_csr_scalar_kernel<float, texReader>(valSP_csr_pad, cols_csr_pad, rowDelimiters_pad, vecSP, dim_pad, outSP);
            #endif
        #elif (FORMAT == 3 || FORMAT == 4)
        // CSR vector
            #if (FORMAT == 3)
            // Normal
                spmv_csr_vector_kernel<float, texReader>(valSP_csr, cols_csr, rowDelimiters, vecSP, dim, outSP);
            #else  
            // Padded
                spmv_csr_vector_kernel<float, texReader>(valSP_csr_pad, cols_csr_pad, rowDelimiters_pad, vecSP, dim_pad, outSP);
            #endif
        #else
        // Ellpackr
    //    spmv_ellpackr_kernel<float, texReader>(valSP_csr, cols_csr, rowDelimiters, vecSP, dim, outSP);

           spmv_ellpackr_kernel<float, texReader>(valSP_ellpackr, cols_ellpackr, rowLengths, vecSP, dim, outSP);
        #endif
    #else 
    // Double precision
        #if (FORMAT == 1 || FORMAT == 2)
        // CSR scalar
            #if (FORMAT == 1) 
            // Normal
                spmv_csr_scalar_kernel<double, texReader>(valDP_csr, cols_csr, rowDelimiters, vecDP, dim, outDP);
            #else 
            // Padded
                spmv_csr_scalar_kernel<double, texReader>(valDP_csr_pad, cols_csr_pad, rowDelimiters_pad, vecDP, dim_pad, outDP);
            #endif
        #elif (FORMAT == 3 || FORMAT == 4)
        // CSR vector
            #if (FORMAT == 3)
            // Normal
                spmv_csr_vector_kernel<double, texReader>(valDP_csr, cols_csr, rowDelimiters, vecDP, dim, outDP);
            #else  
            // Padded
                spmv_csr_vector_kernel<double, texReader>(valDP_csr_pad, cols_csr_pad, rowDelimiters_pad, vecDP, dim_pad, outDP);
            #endif
        #else
        // Ellpackr
            spmv_ellpackr_kernel<double, texReader>(valDP_ellpackr, cols_ellpackr, rowLengths, vecDP, dim, outDP);
        #endif
    #endif
}

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
template <typename fpType, typename texReader>
__device__ void
spmv_csr_scalar_kernel(const fpType * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       fpType * vecSP,
                       const int dim, fpType * __restrict__ out)
{
    int myRow = blockIdx.x * blockDim.x + threadIdx.x;
    //texReader vecTexReader;

    if (myRow < dim)
    {
        fpType t = 0.0f;
        int start = rowDelimiters[myRow];
        int end = rowDelimiters[myRow+1];

        for (int j = start; j < end; j++)
        {
            int col = cols[j];
            t += val[j] * vecSP[col];//vecTexReader(col);
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
template <typename fpType, typename texReader>
__device__ void
spmv_csr_vector_kernel(const fpType * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       fpType * vec,
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
    //texReader vecTexReader;
    __shared__ volatile fpType partialSums[128];

    if (myRow < dim) {
        int warpStart = rowDelimiters[myRow];
        int warpEnd = rowDelimiters[myRow+1];
        fpType mySum = 0;
        for (int j = warpStart + id; j < warpEnd; j += warpSize)
        {
            int col = cols[j];
            mySum += val[j] * vec[col]; //vecTexReader(col);
        }
        partialSums[t] = mySum;

        // Reduce partial sums
        if (id < 16) {
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
__device__ void
spmv_ellpackr_kernel(const fpType * __restrict__ val,
                     const int    * __restrict__ cols,
                     const int    * __restrict__ rowLengths,
                     fpType * vec,
                     const int dim, fpType * __restrict__ out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    //texReader vecTexReader;

    if (t < dim)
    {
        fpType result = 0.0f;
        int max = rowLengths[t];

        for (int i = 0; i < max; i++)
        {
            int ind = i*dim+t;
            result += val[ind] * vec[cols[ind]];//vecTexReader(cols[ind]);
        }
        out[t] = result;
    }
}

