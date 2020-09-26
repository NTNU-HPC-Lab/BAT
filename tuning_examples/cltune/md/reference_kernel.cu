// Uncomment which precision to use for testing, as CLTune does not provide the compiler directives
// Select which precision that are used in the calculations
// #if PRECISION == 32
    #define T float
    #define forceVecType float3
    #define posVecType float4
    #define texReader texReader_sp
// #elif PRECISION == 64
    #define T double
    #define forceVecType double3
    #define posVecType double4
    #define texReader texReader_dp
// #endif

// CLTune does not support texture memory, so this is always disabled
#define useTexture false

// Texture caches for position info
texture<float4, 1, cudaReadModeElementType> posTexture;
texture<int4, 1, cudaReadModeElementType> posTexture_dp;

struct texReader_sp {
   __device__ __forceinline__ float4 operator()(int idx) const
   {
       return tex1Dfetch(posTexture, idx);
   }
};

// CUDA doesn't support double4 textures, so we have to do some conversion
// here, resulting in a bit of overhead, but it's still faster than
// an uncoalesced read
struct texReader_dp {
   __device__ __forceinline__ double4 operator()(int idx) const
   {
#if (__CUDA_ARCH__ < 130)
       // Devices before arch 130 don't support DP, and having the
       // __hiloint2double() intrinsic will cause compilation to fail.
       // This return statement added as a workaround -- it will compile,
       // but since the arch doesn't support DP, it will never be called
       return make_double4(0., 0., 0., 0.);
#else
       int4 v = tex1Dfetch(posTexture_dp, idx*2);
       double2 a = make_double2(__hiloint2double(v.y, v.x),
                                __hiloint2double(v.w, v.z));

       v = tex1Dfetch(posTexture_dp, idx*2 + 1);
       double2 b = make_double2(__hiloint2double(v.y, v.x),
                                __hiloint2double(v.w, v.z));

       return make_double4(a.x, a.y, b.x, b.y);
#endif
   }
};

// ****************************************************************************
// Function: compute_lj_force
//
// Purpose:
//   GPU kernel to calculate Lennard Jones force
//
// Arguments:
//      force3:     array to store the calculated forces
//      position:   positions of atoms
//      neighCount: number of neighbors for each atom to consider
//      neighList:  atom neighbor list
//      cutsq:      cutoff distance squared
//      lj1, lj2:   LJ force constants
//      inum:       total number of atoms
//
// Returns: nothing
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ****************************************************************************
extern "C" __global__ void compute_lj_force(forceVecType* __restrict__ force3,
                                 const posVecType* __restrict__ position,
                                 const int neighCount,
                                 const int* __restrict__ neighList,
                                 const T cutsq,
                                 const T lj1,
                                 const T lj2,
                                 const int inum)
{
    // Global ID - one thread per atom
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // Ensure that the current thread id is less than total number of elements
    if (idx < inum) {
        // Position of this thread's atom
        posVecType ipos = position[idx];

        // Force accumulator
        forceVecType f = {0.0f, 0.0f, 0.0f};

        texReader positionTexReader;

        int j = 0;
        while (j < neighCount)
        {
            int jidx = neighList[j*inum + idx];
            posVecType jpos;
            if (useTexture)
            {
                // Use texture mem as a cache
                jpos = positionTexReader(jidx);
            }
            else
            {
                jpos = position[jidx];
            }

            // Calculate distance
            T delx = ipos.x - jpos.x;
            T dely = ipos.y - jpos.y;
            T delz = ipos.z - jpos.z;
            T r2inv = delx*delx + dely*dely + delz*delz;

            // If distance is less than cutoff, calculate force
            // and add to accumulator
            if (r2inv < cutsq)
            {
                r2inv = 1.0f/r2inv;
                T r6inv = r2inv * r2inv * r2inv;
                T force = r2inv*r6inv*(lj1*r6inv - lj2);

                f.x += delx * force;
                f.y += dely * force;
                f.z += delz * force;
            }
            j++;
        }
        
        // store the results
        force3[idx] = f;
    }
}

extern "C" __global__ void md_helper(
    float3* __restrict__ force3f,
    const float4* __restrict__ positionf,
    double3* __restrict__ force3d,
    const double4* __restrict__ positiond,
    const int neighCount,
    const int* __restrict__ neighList,
    const T cutsq,
    const T lj1,
    const T lj2,
    const int inum
) {
    // #if PRECISION == 32
        compute_lj_force(force3f, positionf, neighCount, neighList, cutsq, lj1, lj2, inum);
    // #elif PRECISION == 64
        compute_lj_force(force3d, positiond, neighCount, neighList, cutsq, lj1, lj2, inum);
    // #endif
}
