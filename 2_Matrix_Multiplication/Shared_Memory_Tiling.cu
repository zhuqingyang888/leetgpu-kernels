#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matrix_multiplication_kernel_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    float sum = 0;

    // N 维度被切成多个 TILE_SIZE 大小的 tile
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for(int tile = 0; tile < numTiles; tile++) {

        // ------------------------------------------------
        // 1. Global Memory -> Shared Memory
        // ------------------------------------------------

        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;

        if(row < M && a_col < N) {
            sharedA[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        }
        else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(b_row < N && col < K){
            sharedB[threadIdx.y][threadIdx.x] = B[b_row * K + col];
        }
        else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 确保整个 tile 已经加载完成
        __syncthreads();

        #pragma unroll
        for(int i = 0; i < TILE_SIZE; i++) {
            sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
        }

        // 当前 tile 使用完之后，
        // 才允许下一轮覆盖 sharedA/sharedB
        __syncthreads();
    }

    if(row < M && col < K) {
        C[row * K + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((K + TILE_SIZE - 1) / TILE_SIZE,
                       (M + TILE_SIZE - 1) / TILE_SIZE);

    matrix_multiplication_kernel_tiled<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    // cudaDeviceSynchronize();
}