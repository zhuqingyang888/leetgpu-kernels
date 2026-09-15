#include <cuda_runtime.h>


__global__ void vector_add_grid_stride(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N){
    int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < N; i += stride){
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve_grid_stride(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int SM_Count = 26; //5060 Laptop SM = 26;
    int MaxthreadsperSM = 1536;
    int blocksPerGrid = 26 * 1536 / threadsPerBlock;

    vector_add_grid_stride<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    // cudaDeviceSynchronize();
}