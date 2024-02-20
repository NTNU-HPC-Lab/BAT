#include <iostream>
#include <cuda_runtime.h>
#include "stdio.h"

#define WIDTH 2048
#define block_size_x 32
#define block_size_y 8
#define tile_size_x 4
#define tile_size_y 4

// Forward declaration of the CUDA kernel
__global__ void matmul_kernel(float *C, float *A, float *B);

// Utility function to check for CUDA errors
void checkCudaErrors(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

// Main program
int main() {
    size_t size = WIDTH * WIDTH * sizeof(float);

    // Allocate and initialize matrices A, B, and C
    float *A, *B, *C;
    float *d_A, *d_B, *d_C; // Device copies of A, B, C

    A = (float *)malloc(size);
    B = (float *)malloc(size);
    C = (float *)malloc(size);

    // Initialize A and B with some values
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        A[i] = 1.0f; // Some value
        B[i] = 2.0f; // Some value
    }

    // Allocate space for device copies of A, B, C
    checkCudaErrors(cudaMalloc((void **)&d_A, size));
    checkCudaErrors(cudaMalloc((void **)&d_B, size));
    checkCudaErrors(cudaMalloc((void **)&d_C, size));

    // Copy inputs to device
    checkCudaErrors(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    // Create events for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));


    // Launch matmul_kernel() on the GPU
    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid((WIDTH + block_size_x * tile_size_x - 1) / (block_size_x * tile_size_x),
             (WIDTH + block_size_y * tile_size_y - 1) / (block_size_y * tile_size_y));
    matmul_kernel<<<dimGrid, dimBlock>>>(d_C, d_A, d_B);

     // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));

    // Check for any errors launching the kernel
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

    // Calculate the elapsed time in milliseconds
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    // Compute and print the performance
    double flops = 2.0 * static_cast<double>(WIDTH) * static_cast<double>(WIDTH) * static_cast<double>(WIDTH);
    double gflops = flops / (milliseconds / 1000.0) / 1e9;
    std::cout << "Performance: " << gflops << " GFLOPs" << std::endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);

    return 0;
}

