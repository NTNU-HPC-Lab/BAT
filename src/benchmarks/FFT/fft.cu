// Written by Vasily Volkov.
// Copyright (c) 2008-2009, The Regents of the University of California.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//    - Redistributions of source code must retain the above copyright notice,
//        this list of conditions and the following disclaimer.
//    - Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer
//        in the documentation and/or other materials provided with the
//        distribution.
//    - Neither the name of the University of California, Berkeley nor the
//        names of its contributors may be used to endorse or promote
//        products derived from this software without specific prior
//        written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include <math.h>
#define M_SQRT1_2 0.70710678118654752440
// #define _USE_MATH_DEFINES
extern "C"{

// #include <cupy/math.cuh>
// #include <cmath>



//
// arrange blocks into 2D grid that fits into the GPU ( for powers of two only )
//
/*
inline dim3 grid2D( int nblocks )
{
    int slices = 1;
    while( nblocks/slices > 65535 )
        slices *= 2;
    return dim3( nblocks/slices, slices );
}
*/
#if PRECISION == 32
#define T float
#define T2 float2
#else
#define T double
#define T2 double2
#endif
/*
template<class T2, class T> __device__ T2 make_T2(T x, T y);

template <> inline __device__
*/
inline __device__
T2 make_T2(T x, T y)
{
#if PRECISION == 32
    return make_float2(x, y);
#else
    return make_double2(x, y);
#endif
}
/*float2 make_T2<float2,float>(float x, float y)
{
    return make_float2(x, y);
}*/

//template <> inline __device__
/*double2 make_T2<double2,double>(double x, double y)
{
    return make_double2(x, y);
}
*/

//
// complex number arithmetic
//
// cmccurdy: unfortunately operator overloading doesn't appear to work
// w/ templates, reverting to explicit functions.

//template <class T2, class T> inline __device__
//T2 operator*( T2 a, T2 b ) { return make_T2<T2,T>( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
//template <class T2, class T>
inline __device__
T2 cmul( T2 a, T2 b ) { return make_T2( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
//template <class T2, class T> inline __device__
//T2 operator*( T2 a, T  b ) { return make_T2<T2,T>( b*a.x, b*a.y ); }
//template <class T2, class T>
inline __device__
T2 cmul_s( T2 a, T  b ) { return make_T2( b*a.x, b*a.y ); }
//template <class T2, class T>
//inline __device__
//T2 operator+( T2 a, T2 b ) { return make_T2<T2,T>( a.x + b.x, a.y + b.y ); }
//template <class T2, class T>
inline __device__
T2 cadd( T2 a, T2 b ) { return make_T2( a.x + b.x, a.y + b.y ); }
//template <class T2, class T>
//inline __device__
//T2 operator-( T2 a, T2 b ) { return make_T2<T2,T>( a.x - b.x, a.y - b.y ); }
//template <class T2, class T>
inline __device__
T2 csub( T2 a, T2 b ) { return make_T2( a.x - b.x, a.y - b.y ); }

#define COS_PI_8  0.923879533f
#define SIN_PI_8  0.382683432f
#define exp_1_16  make_T2(  COS_PI_8, -SIN_PI_8 )
#define exp_3_16  make_T2(  SIN_PI_8, -COS_PI_8 )
#define exp_5_16  make_T2( -SIN_PI_8, -COS_PI_8 )
#define exp_7_16  make_T2( -COS_PI_8, -SIN_PI_8 )
#define exp_9_16  make_T2( -COS_PI_8,  SIN_PI_8 )
#define exp_1_8   make_T2(  1, -1 )//requires post-multiply by 1/sqrt(2)
#define exp_1_4   make_T2(  0, -1 )
#define exp_3_8   make_T2( -1, -1 )//requires post-multiply by 1/sqrt(2)

#define iexp_1_16  make_T2(  COS_PI_8,  SIN_PI_8 )
#define iexp_3_16  make_T2(  SIN_PI_8,  COS_PI_8 )
#define iexp_5_16  make_T2( -SIN_PI_8,  COS_PI_8 )
#define iexp_7_16  make_T2( -COS_PI_8,  SIN_PI_8 )
#define iexp_9_16  make_T2( -COS_PI_8, -SIN_PI_8 )
#define iexp_1_8   make_T2(  1, 1 )//requires post-multiply by 1/sqrt(2)
#define iexp_1_4   make_T2(  0, 1 )
#define iexp_3_8   make_T2( -1, 1 )//requires post-multiply by 1/sqrt(2)

// template <class T2, class T>
inline __device__
T2 exp_i( T phi )
{
    return make_T2( __cosf(phi), __sinf(phi) );
}

//
//  bit reversal
//
inline __device__ int rev_2( int bits )
{
    return bits;
}

inline __device__ int rev_4( int bits )
{
    int reversed[] = {0,2,1,3};
    return reversed[bits];
}

inline __device__ int rev_8( int bits )
{
    int reversed[] = {0,4,2,6,1,5,3,7};
    return reversed[bits];
}

inline __device__ int rev_16( int bits )
{
    int reversed[] = {0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    return reversed[bits];
}

inline
__device__ int rev4x4( int bits )
{
    int reversed[] = {0,2,1,3, 4,6,5,7, 8,10,9,11, 12,14,13,15};
    return reversed[bits];
}

//
//  all FFTs produce output in bit-reversed order
//
#define IFFT2 FFT2
// template<class T2, class T>
inline __device__
void FFT2( T2 &a0, T2 &a1 )
{
    T2 c0 = a0;
//    a0 = c0 + a1;
    a0 = cadd(c0, a1);
//    a1 = c0 - a1;
    a1 = csub(c0, a1);
}

// template<class T2, class T>
inline __device__
void FFT4( T2 &a0, T2 &a1, T2 &a2, T2 &a3 )
{
    FFT2( a0, a2 );
    FFT2( a1, a3 );
//    a3 = a3 * exp_1_4;
    a3 = cmul(a3, exp_1_4);
    FFT2( a0, a1 );
    FFT2( a2, a3 );
}

// template<class T2, class T>
inline __device__
void IFFT4( T2 &a0, T2 &a1, T2 &a2, T2 &a3 )
{
    IFFT2( a0, a2 );
    IFFT2( a1, a3 );
//    a3 = a3 * iexp_1_4;
    a3 = cmul(a3, iexp_1_4);
    IFFT2( a0, a1 );
    IFFT2( a2, a3 );
}

// template<class T2, class T>
//inline __device__
//void FFT2_p( T2 *a ) { FFT2( a[0], a[1] ); }
// template<class T2, class T>
//inline __device__
//void FFT4_p( T2 *a ) { FFT4( a[0], a[1], a[2], a[3] ); }
// template<class T2, class T>
//inline __device__
//void IFFT4_p( T2 *a ) { IFFT4( a[0], a[1], a[2], a[3] ); }

// template<class T2, class T>
inline __device__
void FFT8( T2 *a )
{
    FFT2( a[0], a[4] );
    FFT2( a[1], a[5] );
    FFT2( a[2], a[6] );
    FFT2( a[3], a[7] );

//    a[5] = ( a[5] * exp_1_8 ) * M_SQRT1_2;
//    a[6] =   a[6] * exp_1_4;
//    a[7] = ( a[7] * exp_3_8 ) * M_SQRT1_2;
    a[5] = cmul_s(cmul(a[5], exp_1_8 ), M_SQRT1_2);
    a[6] = cmul(a[6], exp_1_4);
    a[7] = cmul_s(cmul(a[7], exp_3_8 ), M_SQRT1_2);

    FFT4( a[0], a[1], a[2], a[3] );
    FFT4( a[4], a[5], a[6], a[7] );
}

//template<class T2, class T>
inline __device__
void IFFT8( T2 *a )
{
    IFFT2( a[0], a[4] );
    IFFT2( a[1], a[5] );
    IFFT2( a[2], a[6] );
    IFFT2( a[3], a[7] );

//    a[5] = ( a[5] * iexp_1_8 ) * M_SQRT1_2;
//    a[6] =   a[6] * iexp_1_4;
//    a[7] = ( a[7] * iexp_3_8 ) * M_SQRT1_2;
    a[5] = cmul_s(cmul(a[5], iexp_1_8 ), M_SQRT1_2);
    a[6] = cmul(a[6], iexp_1_4);
    a[7] = cmul_s(cmul(a[7], iexp_3_8 ), M_SQRT1_2);

    IFFT4( a[0], a[1], a[2], a[3] );
    IFFT4( a[4], a[5], a[6], a[7] );
}


//
//  loads
//
// template<int n, class T2>
inline __device__
void load_8( T2 *a, T2 *x, int sx )
{
    for( int i = 0; i < 8; i++ )
        a[i] = x[i*sx];
}
// template<int n, class T2, class T>
inline __device__
void loadx_8( T2 *a, T *x, int sx )
{
    for( int i = 0; i < 8; i++ ) {
        a[i].x = x[i*sx];
    }
}
inline __device__
void loadx_16( T2 *a, T *x, int sx )
{
    for( int i = 0; i < 16; i++ )
        a[i].x = x[i*sx];
}

inline __device__
void loady_8( T2 *a, T *x, int sx )
{
    for( int i = 0; i < 8; i++ )
        a[i].y = x[i*sx];
}
inline __device__
void loady_16( T2 *a, T *x, int sx )
{
    for( int i = 0; i < 16; i++ )
        a[i].y = x[i*sx];
}
inline __device__
void loadx_p_8( T2 *a, T *x, int *ind )
{
    for( int i = 0; i < 8; i++ )
        a[i].x = x[ind[i]];
}
inline __device__
void loadx_p_16( T2 *a, T *x, int *ind )
{
    for( int i = 0; i < 16; i++ )
        a[i].x = x[ind[i]];
}
inline __device__
void loady_p_8( T2 *a, T *x, int *ind )
{
    for( int i = 0; i < 8; i++ )
        a[i].y = x[ind[i]];
}
inline __device__
void loady_p_16( T2 *a, T *x, int *ind )
{
    for( int i = 0; i < 16; i++ )
        a[i].y = x[ind[i]];
}

//
//  stores, input is in bit reversed order
//
inline __device__
void store_8( T2 *a, T2 *x, int sx )
{
#pragma unroll
    for( int i = 0; i < 8; i++ )
        x[i*sx] = a[rev_8(i)];
}
inline __device__
void storex_8( T2 *a, T *x, int sx )
{
#pragma unroll
    for( int i = 0; i < 8; i++ )
        x[i*sx] = a[rev_8(i)].x;
}
inline __device__
void storey_8( T2 *a, T *x, int sx )
{
#pragma unroll
    for( int i = 0; i < 8; i++ )
        x[i*sx] = a[rev_8(i)].y;
}
//template<class T2, class T>
inline __device__
void storex4x4( T2 *a, T *x, int sx )
{
#pragma unroll
    for( int i = 0; i < 16; i++ )
        x[i*sx] = a[rev4x4(i)].x;
}
// template<class T2, class T>
inline __device__
void storey4x4( T2 *a, T *x, int sx )
{
#pragma unroll
    for( int i = 0; i < 16; i++ )
        x[i*sx] = a[rev4x4(i)].y;
}

//
//  multiply by twiddle factors in bit-reversed order
//
inline __device__
void twiddle_8( T2 *a, int i, int n )
{
#pragma unroll
    for( int j = 1; j < 8; j++ )
//        a[j] = a[j] * exp_i<T2,T>((-2*M_PI*rev<radix>(j)/n)*i);
        a[j] = cmul(a[j], exp_i((-2*M_PI*rev_8(j)/n)*i));
}

inline __device__
void itwiddle_8( T2 *a, int i, int n )
{
#pragma unroll
    for( int j = 1; j < 8; j++ )
//        a[j] = a[j] * exp_i<T2,T>((2*M_PI*rev<radix>(j)/n)*i);
        a[j] = cmul(a[j], exp_i((2*M_PI*rev_8(j)/n)*i));
}

//
//  transpose via shared memory, input is in bit-reversed layout
//
inline __device__
void transpose_8( T2 *a, T *s, int ds, T *l, int dl, int sync = 0xf )
{
    storex_8( a, s, ds );  if( sync&8 ) __syncthreads();
    loadx_8 ( a, l, dl );  if( sync&4 ) __syncthreads();

    storey_8( a, s, ds );  if( sync&2 ) __syncthreads();
    loady_8 ( a, l, dl );  if( sync&1 ) __syncthreads();
}
/*
inline __device__
void transpose_p_8( T2 *a, T *s, int ds, T *l, int *il, int sync = 0xf )
{
    storex_8( a, s, ds );  if( sync&8 ) __syncthreads();
    loadx_p_8 ( a, l, il );  if( sync&4 ) __syncthreads();
    storey_8( a, s, ds );  if( sync&2 ) __syncthreads();
    loady_p_8 ( a, l, il );  if( sync&1 ) __syncthreads();
}
*/

// template<class T2, class T>
inline __device__
void transpose4x4( T2 *a, T *s, int ds, T *l, int dl, int sync = 0xf )
{
    storex4x4( a, s, ds ); if( sync&8 ) __syncthreads();
    loadx_16( a, l, dl );  if( sync&4 ) __syncthreads();
    storey4x4( a, s, ds ); if( sync&2 ) __syncthreads();
    loady_16( a, l, dl );  if( sync&1 ) __syncthreads();
}

// template<class T2, class T>
__global__
void FFT512_device( T2 *work )
{
    int tid = threadIdx.x;
    int hi = tid>>3;
    int lo = tid&7;

    work += (blockIdx.y * gridDim.x + blockIdx.x) * 512 + tid;

    T2 a[8];
    __shared__ T smem[8*8*9];
    load_8( a, work, 64 );

    FFT8( a );

    twiddle_8( a, tid, 512 );
    transpose_8( a, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8 );

    FFT8( a );

    twiddle_8( a, hi, 64);
    transpose_8( a, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE );

    FFT8( a );

    store_8( a, work, 64 );
}

// template<class T2, class T>
__global__
void IFFT512_device( T2 *work )
{
    int i, tid = threadIdx.x;
    int hi = tid>>3;
    int lo = tid&7;

    work += (blockIdx.y * gridDim.x + blockIdx.x) * 512 + tid;

    T2 a[8];
    __shared__ T smem[8*8*9];

    load_8( a, work, 64 );


    IFFT8( a );

    itwiddle_8( a, tid, 512 );
    transpose_8( a, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8 );

    IFFT8( a );

    itwiddle_8( a, hi, 64);
    transpose_8( a, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE );

    IFFT8( a );

    // normalize...
    for (i = 0; i < 8; i++) {
        a[i].x /= 512;
        a[i].y /= 512;
    }
    store_8( a, work, 64 );

}

} // Extern C