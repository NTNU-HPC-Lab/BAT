__device__ double3 operator+(const double3& lhs, const double3& rhs) {
    return make_double3(lhs.x + rhs.x,
                        lhs.y + rhs.y,
                        lhs.z + rhs.z);
}
extern "C" __global__ void sum_kernel(const double3* lhs,
                                            double3*  rhs,
                                            double3* out,
                                            int N) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < N) {
    out[i] = lhs[i] + rhs[0];
  }
}