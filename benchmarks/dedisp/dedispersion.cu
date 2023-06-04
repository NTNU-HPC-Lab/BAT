
#include "dedispersion.h"


/*

Parallelization:

Grid:
x - samples
y - dms
z - beams, not used now

Block:
x - varies
y - varies
z - 1

*/

extern "C"
void __global__ dedispersion_naive(const unsigned char * input, float * output, const float * shifts) {

    int x = blockIdx.x * block_size_x + threadIdx.x;
    int y = blockIdx.y * block_size_y + threadIdx.y;
    //int z = blockIdx.z * block_size_z + threadIdx.z;

    int sample = x;
    int dm = y;

    if (dm < nr_dms) {
        if (sample < nr_samples) {

            float dedispersedSample = 0.0f;

            for (int channel = 0; channel < nr_channels; channel++) {
                unsigned int shift = (dm_first + dm*dm_step) * shifts[channel];
                dedispersedSample += input[(channel * nr_samples_per_channel) + (sample + shift)];
            }

            output[(dm * nr_samples) + sample] = dedispersedSample;
        }
    }

}


#ifndef tile_size_x
#define tile_size_x 1
#endif
#ifndef tile_size_y
#define tile_size_y 1
#endif


#ifndef tile_stride_x
#define tile_stride_x 1
#endif
#ifndef tile_stride_y
#define tile_stride_y 1
#endif

#if tile_stride_x == 0
#define x_stride 1
#define thread_x_stride tile_size_x
#else
#define x_stride block_size_x
#define thread_x_stride 1
#endif

#if tile_stride_y == 0
#define y_stride 1
#define thread_y_stride tile_size_y
#else
#define y_stride block_size_y
#define thread_y_stride 1
#endif

#ifndef blocks_per_sm
#define blocks_per_sm 0
#endif


extern "C"
__global__ void
#if blocks_per_sm > 0
__launch_bounds__(block_size_x * block_size_y * block_size_z, blocks_per_sm)
#endif
dedispersion_kernel(const unsigned char * input, float * output, const float * shifts) {

    int x = blockIdx.x * block_size_x * tile_size_x + threadIdx.x * thread_x_stride;
    int y = blockIdx.y * block_size_y * tile_size_y + threadIdx.y * thread_y_stride;
    //int z = blockIdx.z * block_size_z + threadIdx.z;

//    #pragma unroll loop_unroll_factor_x
    for (int ti=0; ti < tile_size_x; ti++) {
        int sample = x+ti*x_stride;
        if (sample < nr_samples) {

//            #pragma unroll loop_unroll_factor_y
            for (int tj=0; tj < tile_size_y; tj++) {
                int dm = y+tj*y_stride;

                if (dm < nr_dms) {

                    float dedispersedSample = 0.0f;

                    #pragma unroll loop_unroll_factor_channel
                    for (int channel = 0; channel < nr_channels; channel++) {
                        unsigned int shift = (dm_first + dm*dm_step) * shifts[channel];

                        dedispersedSample += input[(channel * nr_samples_per_channel) + (sample + shift)];
                    }

                    output[(dm * nr_samples) + sample] = dedispersedSample;
                }
            }
        }
    }
}

