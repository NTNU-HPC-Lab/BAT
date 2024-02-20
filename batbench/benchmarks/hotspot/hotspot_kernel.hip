#include "hip/hip_runtime.h"
#ifndef kernel_tuner
  #define grid_width 4096
  #define grid_height 4096
  #define block_size_x 16
  #define block_size_y 16
  #define tile_size_x 1
  #define tile_size_y 1
  #define temporal_tiling_factor 1
  #define max_tfactor 10
  #define sh_power 0
  #define blocks_per_sm 0
  #define loop_unroll_factor_t 1
#endif


//calculate shared memory size, depends on temporal_tiling_factor and on tile_size_x/y
#define tile_width block_size_x*tile_size_x + temporal_tiling_factor * 2
#define tile_height block_size_y*tile_size_y + temporal_tiling_factor * 2


#define amb_temp 80.0f


#define input_width (grid_width+max_tfactor*2)   //could add padding
#define input_height (grid_height+max_tfactor*2)

#define output_width grid_width
#define output_height grid_height

__global__
#if blocks_per_sm > 0
__launch_bounds__(block_size_x * block_size_y * block_size_z, blocks_per_sm)
#endif
void calculate_temp(float *power,          //power input
                               float *temp,           //temperature input
                               float *temp_dst,       //temperature output
                               const float Rx_1,
                               const float Ry_1,
                               const float Rz_1,
                               const float step_div_cap) {

    //offset input pointers to make the code testable with different temporal tiling factors
    float* power_src = power+(max_tfactor-temporal_tiling_factor)*input_width+max_tfactor-temporal_tiling_factor;
    float* temp_src = temp+(max_tfactor-temporal_tiling_factor)*input_width+max_tfactor-temporal_tiling_factor;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int k=1;

    __shared__ float temp_on_cuda[2][tile_height][tile_width]; //could add padding
    #if sh_power == 1
    __shared__ float power_on_cuda[tile_height][tile_width];
    #endif

    // fill shared memory with values
    #pragma unroll
    for (int j=ty; j<tile_height; j+=block_size_y) {
        #pragma unroll
        for (int i=tx; i<tile_width; i+=block_size_x) {
            int x = tile_size_x*block_size_x*blockIdx.x+i;
            int y = tile_size_y*block_size_y*blockIdx.y+j;
            if (x < input_width && y < input_height) {
                temp_on_cuda[k][j][i] = temp_src[y*input_width + x];
    #if sh_power == 1
                power_on_cuda[j][i] = power_src[y*input_width + x];
    #endif
            } else {
                temp_on_cuda[1-k][j][i] = 0.0;
                temp_on_cuda[k][j][i] = 0.0;
    #if sh_power == 1
                power_on_cuda[j][i] = 0.0;
    #endif
            }
        }
    }
    __syncthreads();


    //main computation
    #pragma unroll loop_unroll_factor_t
    for (int iteration=1; iteration <= temporal_tiling_factor; iteration++) {

        //cooperatively compute the area, shrinking with each iteration
        #pragma unroll
        for (int j=ty+iteration; j<tile_height-iteration; j+=block_size_y) {
            int N = j-1;
            int S = j+1;

            #pragma unroll
            for (int i=tx+iteration; i<tile_width-iteration; i+=block_size_x) {
                int W = i-1;
                int E = i+1;

                //do computation
    #if sh_power == 1
                temp_on_cuda[1-k][j][i] = temp_on_cuda[k][j][i] + step_div_cap * (power_on_cuda[j][i] +
                     (temp_on_cuda[k][S][i] + temp_on_cuda[k][N][i] - 2.0*temp_on_cuda[k][j][i]) * Ry_1 +
                     (temp_on_cuda[k][j][E] + temp_on_cuda[k][j][W] - 2.0*temp_on_cuda[k][j][i]) * Rx_1 +
                     (amb_temp - temp_on_cuda[k][j][i]) * Rz_1);
    #else

                int x = tile_size_x*block_size_x*blockIdx.x+i;
                int y = tile_size_y*block_size_y*blockIdx.y+j;
                float power = 0.0f;
                if (x < input_width && y < input_height) {
                    power = power_src[y*input_width + x];
                }

                temp_on_cuda[1-k][j][i] = temp_on_cuda[k][j][i] + step_div_cap * (power +
                     (temp_on_cuda[k][S][i] + temp_on_cuda[k][N][i] - 2.0*temp_on_cuda[k][j][i]) * Ry_1 +
                     (temp_on_cuda[k][j][E] + temp_on_cuda[k][j][W] - 2.0*temp_on_cuda[k][j][i]) * Rx_1 +
                     (amb_temp - temp_on_cuda[k][j][i]) * Rz_1);
    #endif

            }
        }

        __syncthreads();


        //swap

        k = 1-k;
        //for (int j=ty+iteration; j<tile_height-iteration; j+=block_size_y) {
        //    for (int i=tx+iteration; i<tile_width-iteration; i+=block_size_x) {
        //        temp_on_cuda[j][i] = temp_t[j][i];
        //    }
        //}
        //__syncthreads();

    }


    //write out result, should be 1 per thread unless spatial blocking is used
    #pragma unroll
    for (int tj=0; tj<tile_size_y; tj++) {
        #pragma unroll
        for (int ti=0; ti<tile_size_x; ti++) {
            int x = tile_size_x*block_size_x*blockIdx.x+ti*block_size_x+tx;
            int y = tile_size_y*block_size_y*blockIdx.y+tj*block_size_y+ty;
            if (x < output_width && y < output_height) {
                temp_dst[y*output_width + x] = temp_on_cuda[k][tj*block_size_y+ty+temporal_tiling_factor][ti*block_size_x+tx+temporal_tiling_factor];
            }
        }
    }


}
