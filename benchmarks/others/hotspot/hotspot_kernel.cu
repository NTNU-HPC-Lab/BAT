#define kernel_tuner 1
#ifndef kernel_tuner
#define GRID_SIZE_X 4096
#define GRID_SIZE_Y 4096
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define TILE_SIZE_X 1
#define TILE_SIZE_Y 1
#define TEMPORAL_TILING_FACTOR 1
#define MAX_TFACTOR 10
#define SH_POWER 0
#define BLOCKS_PER_SM 0
#endif


//calculate shared memory size, depends on TEMPORAL_TILING_FACTOR and on TILE_SIZE_X/y
#define tile_width BLOCK_SIZE_X*TILE_SIZE_X + TEMPORAL_TILING_FACTOR * 2
#define tile_height BLOCK_SIZE_Y*TILE_SIZE_Y + TEMPORAL_TILING_FACTOR * 2


#define amb_temp 80.0f


#define input_width (GRID_SIZE_X+MAX_TFACTOR*2)   //could add padding
#define input_height (GRID_SIZE_Y+MAX_TFACTOR*2)

#define output_width GRID_SIZE_X
#define output_height GRID_SIZE_Y

__global__
#if BLOCKS_PER_SM > 0
__launch_bounds__(BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z, BLOCKS_PER_SM)
#endif
    void calculate_temp(float *power,   //power input
            float *temp,                //temperature input
            float *temp_dst,            //temperature output
            const float Rx_1,
            const float Ry_1,
            const float Rz_1,
            const
            float
            step_div_cap)
{

    //offset input pointers to make the code testable with different temporal tiling factors
    float* power_src = power+(MAX_TFACTOR-TEMPORAL_TILING_FACTOR)*input_width+MAX_TFACTOR-TEMPORAL_TILING_FACTOR;
    float* temp_src = temp+(MAX_TFACTOR-TEMPORAL_TILING_FACTOR)*input_width+MAX_TFACTOR-TEMPORAL_TILING_FACTOR;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int k=1;

    __shared__ float temp_on_cuda[2][tile_height][tile_width]; //could add padding
#if SH_POWER == 1
    __shared__ float power_on_cuda[tile_height][tile_width];
#endif

    // fill shared memory with values
#pragma unroll
    for (int j=ty; j<tile_height; j+=BLOCK_SIZE_Y) {
#pragma unroll
        for (int i=tx; i<tile_width; i+=BLOCK_SIZE_X) {
            int x = TILE_SIZE_X*BLOCK_SIZE_X*blockIdx.x+i;
            int y = TILE_SIZE_Y*BLOCK_SIZE_Y*blockIdx.y+j;
            if (x < input_width && y < input_height) {
                temp_on_cuda[k][j][i] = temp_src[y*input_width + x];
#if SH_POWER == 1
                power_on_cuda[j][i] = power_src[y*input_width + x];
#endif
            } else {
                temp_on_cuda[1-k][j][i] = 0.0;
                temp_on_cuda[k][j][i] = 0.0;
#if SH_POWER == 1
                power_on_cuda[j][i] = 0.0;
#endif
            }
        }
    }
    __syncthreads();


    //main computation
#pragma unroll LOOP_UNROLL_FACTOR_T
    for (int iteration=1; iteration <= TEMPORAL_TILING_FACTOR; iteration++) {

        //cooperatively compute the area, shrinking with each iteration
#pragma unroll
        for (int j=ty+iteration; j<tile_height-iteration; j+=BLOCK_SIZE_Y) {
            int N = j-1;
            int S = j+1;

#pragma unroll
            for (int i=tx+iteration; i<tile_width-iteration; i+=BLOCK_SIZE_X) {
                int W = i-1;
                int E = i+1;

                //do computation
#if SH_POWER == 1
                temp_on_cuda[1-k][j][i] = temp_on_cuda[k][j][i] + step_div_cap * (power_on_cuda[j][i] +
                        (temp_on_cuda[k][S][i] + temp_on_cuda[k][N][i] - 2.0*temp_on_cuda[k][j][i]) * Ry_1 +
                        (temp_on_cuda[k][j][E] + temp_on_cuda[k][j][W] - 2.0*temp_on_cuda[k][j][i]) * Rx_1 +
                        (amb_temp - temp_on_cuda[k][j][i]) * Rz_1);
#else

                int x = TILE_SIZE_X*BLOCK_SIZE_X*blockIdx.x+i;
                int y = TILE_SIZE_Y*BLOCK_SIZE_Y*blockIdx.y+j;
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
        //for (int j=ty+iteration; j<tile_height-iteration; j+=BLOCK_SIZE_Y) {
        //    for (int i=tx+iteration; i<tile_width-iteration; i+=BLOCK_SIZE_X) {
        //        temp_on_cuda[j][i] = temp_t[j][i];
        //    }
        //}
        //__syncthreads();

    }


    //write out result, should be 1 per thread unless spatial blocking is used
#pragma unroll
    for (int tj=0; tj<TILE_SIZE_Y; tj++) {
#pragma unroll
        for (int ti=0; ti<TILE_SIZE_X; ti++) {
            int x = TILE_SIZE_X*BLOCK_SIZE_X*blockIdx.x+ti*BLOCK_SIZE_X+tx;
            int y = TILE_SIZE_Y*BLOCK_SIZE_Y*blockIdx.y+tj*BLOCK_SIZE_Y+ty;
            if (x < output_width && y < output_height) {
                temp_dst[y*output_width + x] = temp_on_cuda[k][tj*BLOCK_SIZE_Y+ty+TEMPORAL_TILING_FACTOR][ti*BLOCK_SIZE_X+tx+TEMPORAL_TILING_FACTOR];
            }
        }
    }
}
