/**
 * The kernel is assumed to be tuned to each device by selecting
 * the best performing combination of thread block dimensions
 * and tiling factors in X and Y. In this implementation tiling
 * in X increases the amount of work per thread block and tiling
 * in Y increases the amount of work per thread within the block.
 *
 * WARNING: THIS KERNEL IS FOR EDUCATIONAL PURPOSES ONLY.
 *          PLEASE *DO NOT USE IT* IN PRODUCTION, USE A BLAS
 *          LIBRARY SUCH AS CUBLAS, CLBLAST OR CUTLASS INSTEAD.
 *
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 */

#define WIDTH 2048
#define block_size_x 32
#define block_size_y 8
#define tile_size_x 4
#define tile_size_y 4
#include "stdio.h"

/*
 * Optimized CUDA kernel for matrix multiplication
 *
 * This kernel is optimized according to the directions given
 * in: "Better performance at lower occupancy" by V. Volkov,
 * GPU Technology Conference, GTC 2010.
 *
 * The thread block dimensions (block_size_x, block_size_y)
 * and tiling factors (tile_size_x, tile_size_y) are to be
 * tuned towards each GPU. This kernel assumes that
 * block_size_x = block_size_y * tile_size_y.
 *
 * The kernel computes C=A*B, where A, B, and C are square
 * matrices with height and width equal to WIDTH
 */
__global__ void matmul_kernel(float *C, float *A, float *B) {

    __shared__ float sA[block_size_y*tile_size_y][block_size_x];
    __shared__ float sB[block_size_y*tile_size_y][block_size_x * tile_size_x];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * block_size_x * tile_size_x + threadIdx.x;
    int y = blockIdx.y * block_size_y * tile_size_y + threadIdx.y;
    int k, kb;

    if (x >= WIDTH || y >= WIDTH) {
        printf("Index out of bounds for global memory access: x = %d, y = %d\n", x, y);
        return; // Skip threads with out-of-bounds global indices
    }

    float sum[tile_size_y][tile_size_x] = {0.0f};

    for (k = 0; k < WIDTH; k += block_size_x) {
        __syncthreads();

        // Load A and B matrices into shared memory
        // Ensure the indices are within bounds for sA and sB
        if (ty < block_size_y * tile_size_y && tx < block_size_x && y < WIDTH && k + tx < WIDTH) {
            sA[ty][tx] = A[y * WIDTH + k + tx];
        }
        if (ty < block_size_y * tile_size_y && tx < block_size_x * tile_size_x && k + ty < WIDTH && x < WIDTH) {
            sB[ty][tx] = B[(k + ty) * WIDTH + x];
        }

        __syncthreads();

        //compute
        for (kb = 0; kb < block_size_x; kb++) {
            for (int i = 0; i < tile_size_y; i++) {
                for (int j = 0; j < tile_size_x; j++) {
                    if (ty + i * block_size_y < block_size_y * tile_size_y &&
                        tx + j * block_size_x < block_size_x * tile_size_x) {
                        sum[i][j] += sA[ty + i * block_size_y][kb] * sB[kb][tx + j * block_size_x];
                    //} else {
                        //printf("Index out of bounds in computation: ty = %d, i = %d, kb = %d, tx = %d, j = %d\n",
                        //       ty, i, kb, tx, j);
                    }

                }
            }
        }

    }

    // Store result
    for (int i = 0; i < tile_size_y; ++i) {
        for (int j = 0; j < tile_size_x; ++j) {
            if (y + i * block_size_y < WIDTH && x + j * block_size_x < WIDTH) {
                C[(y + i * block_size_y) * WIDTH + x + j * block_size_x] = sum[i][j];
            } else {
                printf("Index out of bounds for global memory access: y = %d, i = %d, x = %d, j = %d\n", y, i, x, j);
            }
        }
    }

    //store result
    /*
    #pragma unroll
    for (int i = 0; i < tile_size_y; i++) {
        #pragma unroll
        for (int j = 0; j < tile_size_x; j++) {
            C[y * WIDTH + x + block_size_y * i * WIDTH + j * block_size_x] = sum[i][j];
        }
    }
    */
}



