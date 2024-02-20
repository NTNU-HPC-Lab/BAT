#include <iostream>
#include <cuda_runtime.h>
#include <random>

#define MWG 64
#define NWG 64
#define KWG 32
#define MDIMC 32
#define NDIMC 32
#define MDIMA 16
#define NDIMB 16
#define KWI 2
#define VWM 1
#define VWN 1
#define STRM 0
#define STRN 0
#define SA 0
#define SB 0
#define PRECISION 32
#include "gemm.cu"  // Include the CUDA kernel

// Utility function for CUDA error checking
void checkCudaErrors(cudaError_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
      file << ":" << line << " '" << func << "' \n";
    cudaDeviceReset();
    exit(99);
  }
}
#define CUDA_CHECK(val) checkCudaErrors((val), #val, __FILE__, __LINE__)

// Function to verify the correctness of the computation
bool verifyResult(float* h_C, int M, int N) {
  bool isCorrect = true;
  for (int i = 0; i < M * N; ++i) {
    if (h_C[i] == 0.0f) { // or any other condition you expect
      isCorrect = false;
      printf("i = %d, h_C[i] = %f\n", i, h_C[i]);
      break;
    }
  }
  return isCorrect;
}

// Main function
int main() {
  // Set kernel configuration parameters

  // Matrix dimensions
  const int M = 2048;
  const int N = 2048;
  const int K = 2048;

  // Allocate host matrices A, B, and C
  float *h_A = new float[M * K];
  float *h_B = new float[K * N];
  float *h_C = new float[M * N];

  // Random number generation setup
  std::mt19937 eng(42); // Use a constant value for the seed
  std::uniform_real_distribution<> distr(0.0, 1.0); // Define the range

  // Initialize host matrices A and B with random values
  for (int i = 0; i < M * K; ++i) h_A[i] = distr(eng);
  for (int i = 0; i < K * N; ++i) h_B[i] = distr(eng);

  // Allocate device matrices A, B, and C
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

  // Copy host matrices A and B to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Define kernel execution configuration
  dim3 blockDim(MDIMC, NDIMC, 1);
  dim3 gridDim((M / MDIMC) / (MWG/MDIMC), (N / NDIMC) / (NWG/NDIMC), 1);

  // Record the start event
  CUDA_CHECK(cudaEventRecord(start, NULL));

  // Launch the kernel
  gemm_fast<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);


  // Record the end event
  CUDA_CHECK(cudaEventRecord(stop, NULL));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  // Calculate and print GFLOP/s
  double gflops = (2.0 * M * N * K) / (milliseconds / 1000.0) / 1e9;
  std::cout << "GFLOP/s: " << gflops << std::endl;

  // Copy the result matrix C from device to host if needed
  CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify the result
  if (verifyResult(h_C, M, N)) {
    std::cout << "Verification passed: All elements in matrix C have been modified." << std::endl;
  } else {
    std::cout << "Verification failed: Some elements in matrix C are not as expected." << std::endl;
  }

  // Free memory
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  // Free host memory
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  // Clean up events
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}

