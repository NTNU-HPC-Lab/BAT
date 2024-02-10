#include "hip/hip_runtime.h"

// =================================================================================================
// This file is CUDA adaptation of a part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains an example OpenCL kernel as part of the gemm.cc example. It is an optimized
// matrix-multiplication kernel according to the paper by Matsumoto et al. and the tutorial on
// http://www.cedricnugteren.nl/tutorial.php. It is fully configurable (and tunable!) using more or
// less the same parameters/naming conventions as in the paper. It supports single and double
// precision (SGEMM/DGEMM) through a pre-processor define.
//
// Note: this kernel requires a compiler compliant to OpenCL 1.1 or higher.
//
// -------------------------------------------------------------------------------------------------
//
// Copyright 2014 SURFsara
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//  http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// =================================================================================================
//
// Matrices are accessed as follows:
// A: [k*M + m], with 'k' ranging from 0:K and 'm' from 0:M (m,k,m)
// B: [k*N + n], with 'k' ranging from 0:K and 'n' from 0:N (n,k,n)
// C: [n*M + m], with 'n' ranging from 0:N and 'm' from 0:M (m,n,m)
//
// Or as an image (assuming column-major)
//       K                      
//    o-------o                 
//    |       |                 
//  N | [B^T] |                 
//    |       |                 
//    o-------o                 
//        K               N     
//    o-------o        o-----o  
//  M |  [A]  |      M | [C] |  
//    |       |        |     |  
//    o-------o        o-----o  
//                              
//
// Parameters determined by the tuner
// MWG       : Tile-size in dimension M (e.g. 64, 128)
// NWG       : Tile-size in dimension N (e.g. 64, 128)
// KWG       : Tile-size in dimension K (e.g. 8, 16)
// MDIMC     : Threads per workgroup in M-dimension (e.g. 8, 16, 32)
// NDIMC     : Threads per workgroup in N-dimension (e.g. 8, 16, 32)
// MDIMA     : Re-shaped tile dimension of matrix A: KDIMA * MDIMA
// NDIMB     : Re-shaped tile dimension of matrix B: KDIMB * NDIMB
// KWI       : Unroll factor of the KWG loop (smaller or equal than KWG)
// VWM       : Vector width of matrices A and C (supported 1, 2, 4, and 8)
// VWN       : Vector width of matrix B (supported 1, 2, 4, and 8)
// STRM      : Use strided access within a thread in the M-dimension (1) or not (0)
// STRN      : Use strided access within a thread in the N-dimension (1) or not (0)
// SA        : Use local/shared memory to cache matrix A (1) or not (0)
// SB        : Use local/shared memory to cache matrix B (1) or not (0)
// PRECISION : Whether to use single (32) or double (64) precision data-types
//
// =================================================================================================

// Helper parameters based on the above tuning parameters
#define MWI (MWG/MDIMC)               // Work per work-item (M-dimension)
#define NWI (NWG/NDIMC)               // Work per work-item (N-dimension)
#define KDIMA ((MDIMC*NDIMC)/(MDIMA)) // Re-shaped tile dimension of matrix A: KDIMA * MDIMA
#define KDIMB ((MDIMC*NDIMC)/(NDIMB)) // Re-shaped tile dimension of matrix B: KDIMB * NDIMB
#define MWA (MWG/MDIMA)               // Amount of loads-per-thread for matrix A (M-dimension)
#define KWA (KWG/KDIMA)               // Amount of loads-per-thread for matrix A (K-dimension)
#define KWB (KWG/KDIMB)               // Amount of loads-per-thread for matrix B (K-dimension)
#define NWB (NWG/NDIMB)               // Amount of loads-per-thread for matrix B (N-dimension)

// Settings
#define USE_VECTOR_MAD 1              // Don't unroll the vector MAD computation
#define USE_CL_MAD 0                  // Uses the non-IEEE754 compliant OpenCL mad() (if above is 0)
#define ZERO 0.0f

extern "C" {  // Needed by CUPY for Python-based tuners

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
#if SA == 1
inline __device__ void GlobalToLocalA(const float* __restrict__ agm, float* alm,
                           const int kSizeM, const int tid, const int kwg) {
  const int la0 = tid % MDIMA;
  const int la1 = tid / MDIMA;
  #pragma unroll
  for (int mia=0; mia<MWA/VWM; ++mia) {
    #pragma unroll
    for (int kia=0; kia<KWA; ++kia) {

      // Computes the indices based on strided/non-strided access
      #if STRM == 0
        int mg = mia + la0*(MWA/VWM);
      #elif STRM == 1
        int mg = la0 + mia*MDIMA;
      #endif

      // Computes the indices for the global memory
      int kg = kia + la1*KWA;
      int idm = mg + blockIdx.x*(MWG/VWM);
      int idk = kg + kwg;

      // Loads the data from global memory (not transposed) into the local memory
      alm[kg*(MWG/VWM) + mg] = agm[idk*(kSizeM/VWM) + idm];
    }
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
inline __device__ void GlobalToLocalB(const  float* __restrict__ bgm, float* blm,
                           const int kSizeN, const int tid, const int kwg) {
  const int lb0 = tid % NDIMB;
  const int lb1 = tid / NDIMB;
  #pragma unroll
  for (int kib=0; kib<KWB; ++kib) {
    #pragma unroll
    for (int nib=0; nib<NWB/VWN; ++nib) {

      // Computes the indices based on strided/non-strided access
      #if STRN == 0
        int ng = nib + lb0*(NWB/VWN);
      #elif STRN == 1
        int ng = lb0 + nib*NDIMB;
      #endif

      // Computes the indices for the global memory
      int kg = kib + lb1*KWB;
      int idn = ng + blockIdx.y*(NWG/VWN);
      int idk = kg + kwg;

      // Loads the data from global memory (transposed) into the local memory
      blm[kg*(NWG/VWN) + ng] = bgm[idk*(kSizeN/VWN) + idn];
    }
  }
}
#endif

// =================================================================================================

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix.
#if SA == 0
inline __device__ void GlobalToPrivateA(const  float* __restrict__ agm, float apm[MWI/VWM],
                             const int kSizeM, const int idk, const int kwg) {
  #pragma unroll
  for (int mi=0; mi<MWI/VWM; ++mi) {

    // Computes the indices based on strided/non-strided access
    #if STRM == 0
      int mg = mi + threadIdx.x*(MWI/VWM);
    #elif STRM == 1
      int mg = threadIdx.x + mi*MDIMC;
    #endif

    // Computes the indices for the global memory
    int idm = mg + blockIdx.x*(MWG/VWM);

    // Loads the data from global memory (not transposed) and stores into registers
    apm[mi] = agm[idk*(kSizeM/VWM) + idm];
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 0
inline __device__ void GlobalToPrivateB(const  float* __restrict__ bgm, float bpm[NWI/VWN],
                             const int kSizeN, const int idk) {
  #pragma unroll
  for (int ni=0; ni<NWI/VWN; ++ni) {

    // Computes the indices based on strided/non-strided access
    #if STRN == 0
      int ng = ni + threadIdx.y*(NWI/VWN);
    #elif STRN == 1
      int ng = threadIdx.y + ni*NDIMC;
    #endif

    // Computes the indices for the global memory
    int idn = ng + blockIdx.y*(NWG/VWN);

    // Loads the data from global memory (transposed) and stores into registers
    bpm[ni] = bgm[idk*(kSizeN/VWN) + idn];
  }
}
#endif

// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
#if SA == 1
inline __device__ void LocalToPrivateA(float* alm, float apm[MWI/VWM], const int kg) {
  #pragma unroll
  for (int mi=0; mi<MWI/VWM; ++mi) {
    #if STRM == 0
      int mg = mi + threadIdx.x*(MWI/VWM);
    #elif STRM == 1
      int mg = threadIdx.x + mi*MDIMC;
    #endif
    apm[mi] = alm[kg*(MWG/VWM) + mg];
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
inline __device__ void LocalToPrivateB(float* blm, float bpm[NWI/VWN], const int kg) {
  #pragma unroll
  for (int ni=0; ni<NWI/VWN; ++ni) {
    #if STRN == 0
      int ng = ni + threadIdx.y*(NWI/VWN);
    #elif STRN == 1
      int ng = threadIdx.y + ni*NDIMC;
    #endif
    bpm[ni] = blm[kg*(NWG/VWN) + ng];
  }
}
#endif

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm
inline __device__ void StoreResults( float* cgm, float cpm[NWI][MWI], const int kSizeM) {
  #pragma unroll
  for (int ni=0; ni<NWI; ++ni) {
    #pragma unroll
    for (int mi=0; mi<MWI; ++mi) {
      /*if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("ni=%d, mi=%d\n", ni, mi);
      }*/
      #if STRM == 0
        int mg = mi + threadIdx.x*MWI;
      #elif STRM == 1
        int mg = threadIdx.x + mi*MDIMC;
      #endif
      #if STRN == 0
        int ng = ni + threadIdx.y*NWI;
      #elif STRN == 1
        int ng = ni + threadIdx.y + ni*NDIMC;
      #endif
      int idm = mg + blockIdx.x*MWG;
      int idn = ng + blockIdx.y*NWG;
      int index = idn*(kSizeM) + idm;
      cgm[index] = cpm[ni][mi];
      //cgm[index] = 1.5f;
    }
  }
}


// =================================================================================================

// The basic scalar multiply-add function
#if USE_CL_MAD == 1
  #define MultiplyAdd(cval, aval, bval) (cval = mad(aval, bval, cval))
#else
  #define MultiplyAdd(cval, aval, bval) (cval += (aval) * (bval))
#endif

// The vectorised multiply-add function
inline __device__ float MultiplyAddVector(float cvec, const float avec, const float bval) {
  cvec += avec * bval;
  return cvec;
}

// Performs the actual computation: Cpm += Apm * Bpm
inline __device__ void MultiplyAccumulate(float cpm[NWI][MWI], float apm[MWI], float bpm[NWI]) {
  #pragma unroll
  for (int ni=0; ni<NWI; ++ni) {
    #pragma unroll
    for (int mi=0; mi<MWI; ++mi) {

        cpm[ni][mi] += apm[mi] * bpm[ni];
        //cpm[ni][mi] += 1.02 * 1.6;
    }
  }
}

// =================================================================================================

// Main entry of the kernel. This function contains the basic skeleton, the functionality is
// provided by the inlined functions above.
__global__ void gemm_fast(int kSizeM, int kSizeN, int kSizeK,
                        float* agm,
                        float* bgm,
                        float* cgm) {
  // Allocates workitem-private memory (registers)
  float apm[MWI];
  float bpm[NWI];
  float cpm[NWI][MWI];
  // Combined thread identifier
  #if SA == 1 || SB == 1
    volatile int tid = threadIdx.x + MDIMC*threadIdx.y;
  #endif

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __shared__ float alm[KWG * MWG];
  #endif
  #if SB == 1
    __shared__ float blm[KWG * NWG];
  #endif

  // Initializes the accumulation registers
  #pragma unroll
  for (int mi=0; mi<MWI; ++mi) {
    #pragma unroll
    for (int ni=0; ni<NWI; ++ni) {
        cpm[ni][mi] = (float)ZERO;
    }
  }

  // Loops over all workgroup tiles
  for (int kwg=0; kwg<kSizeK; kwg+=KWG) {

    // Loads data: off-chip --> local (matrix A)
    #if SA == 1
      GlobalToLocalA(agm, alm, kSizeM, tid, kwg);
    #endif
    // Loads data: off-chip --> local (matrix B)
    #if SB == 1
      GlobalToLocalB(bgm, blm, kSizeN, tid, kwg);
    #endif

    // Synchronizes all threads in a workgroup
    #if SA == 1 || SB == 1
      __syncthreads();
    #endif

    // Loops over all workitem tiles, unrolled by a factor KWI
    for (int pwi=0; pwi<KWG; pwi+=KWI) {
      #pragma unroll
      for (int pit=0; pit<KWI; ++pit) {
        #if SA == 0 || SB == 0
          int idk = kwg + pwi + pit;
        #endif

        #if SA == 1 || SB == 1
          int kg = pwi+pit;
        #endif

        // Loads data: local --> private (matrix A)
        #if SA == 1
          LocalToPrivateA(alm, apm, kg);
        // Loads data: off-chip --> private (matrix A)
        #else
          GlobalToPrivateA(agm, apm, kSizeM, idk, kwg);
        #endif

        // Loads data: local --> private (matrix B)
        #if SB == 1
          LocalToPrivateB(blm, bpm, kg);
        // Loads data: off-chip --> private (matrix B)
        #else
          GlobalToPrivateB(bgm, bpm, kSizeN, idk);
        #endif

        // Performs the accumulation (Cpm += Apm * Bpm)
        MultiplyAccumulate(cpm, apm, bpm);
      }
    }
    // Synchronizes all threads in a workgroup
    #if SA == 1 || SB == 1
      __syncthreads();
    #endif

  }

  // Stores an MWG * NWG tile of results
  StoreResults(cgm, cpm, kSizeM);
}

} // Extern C

// =================================================================================================
