#ifndef SPMV_H_
#define SPMV_H_

#include<cuda.h>

static const int WARP_SIZE = 32;

// Forward declarations for kernels
template <typename fpType, typename texReader>
__global__ void
spmv_csr_scalar_kernel(const fpType * __restrict__ val,
                       const int    * __restrict__ cols,
                       const int    * __restrict__ rowDelimiters,
                       const int dim, fpType * __restrict__ out);

template <typename fpType, typename texReader>
__global__ void
spmv_csr_vector_kernel(const fpType * __restrict__ val,
             	       const int    * __restrict__ cols,
		               const int    * __restrict__ rowDelimiters,
                       const int dim, fpType * __restrict__ out);

template <typename fpType, typename texReader>
__global__ void
spmv_ellpackr_kernel(const fpType * __restrict__ val,
		             const int    * __restrict__ cols,
		             const int    * __restrict__ rowLengths,
                     const int dim, fpType * __restrict__ out);


#endif // SPMV_H_
