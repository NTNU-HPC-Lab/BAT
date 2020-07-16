
// ****************************************************************************
// Function: triad
//
// Purpose:
//   A simple vector addition kernel
//   C = A + s*B
//
// Arguments:
//   A,B - input vectors
//   C - output vectors
//   s - scalar
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
extern "C" __global__ void triad(float* A, float* B, float* C, float s)
{
    int gid = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
    C[gid] = A[gid] + s*B[gid];
}