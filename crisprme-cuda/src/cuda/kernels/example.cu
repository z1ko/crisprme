
/// Set the input array 'data' to all ones
extern "C" __global__ void example(float *data, const size_t n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = 1.0f;
  }
}
