#include <cuda_runtime.h>

#define TILE_M 64   // 一个block在M方向计算64行
#define TILE_N 32   // N方向每轮处理32项  
#define TILE_K 32   // 一个block在方向计算32列
#define TM 16        // 一个线程在M方向处理16个数据

__global__ void matrix_multiplication_kernel_1D_Register(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    
    __shared__ float sharedA[TILE_M][TILE_N];
    __shared__ float sharedB[TILE_N][TILE_K];

    float sum[TM] = {0.0f};

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int numThreads = blockDim.x * blockDim.y;

    int numTiles = (N + TILE_N - 1) / TILE_N;

    for(int tile = 0; tile < numTiles; tile++) {
        // load sharedA
        #pragma unroll
        for(int idx = tid; idx < TILE_M * TILE_N; idx += numThreads) {
            int load_a_row = idx / TILE_N;
            int load_a_col = idx % TILE_N;
            int ga_row = blockIdx.y * TILE_M + load_a_row;
            int ga_col = tile * TILE_N + load_a_col;
            sharedA[load_a_row][load_a_col] = (ga_row < M && ga_col < N) ? A[ga_row * N + ga_col] : 0.0f;
        }

        // load sharedB
        #pragma unroll
        for(int idx = tid; idx < TILE_N * TILE_K; idx += numThreads) {
            int load_b_row = idx / TILE_K;
            int load_b_col = idx % TILE_K;
            int gb_row = tile * TILE_N + load_b_row;
            int gb_col = blockIdx.x * TILE_K + load_b_col;
            sharedB[load_b_row][load_b_col] = (gb_row < N && gb_col < K) ? B[gb_row * K + gb_col] : 0.0f;
        }

        // 等待加载完成
        __syncthreads();

        #pragma unroll
        for(int n = 0; n < TILE_N; n++) {
            float b = sharedB[n][threadIdx.x];
            #pragma unroll
            for(int m = 0; m < TM; m++) {
                sum[m] += sharedA[threadIdx.y * TM + m][n] * b;
            }
        }

        //等待计算完成
        __syncthreads();
    }

    #pragma unroll
    for(int i = 0; i < TM; i++) {
        int gc_row = blockIdx.y * TILE_M + threadIdx.y * TM + i;
        int gc_col = blockIdx.x * TILE_K + threadIdx.x;

        if(gc_row < M && gc_col < K) {
            C[gc_row * K + gc_col] = sum[i];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve_1D_Register(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_K, TILE_M / TM);
    dim3 blocksPerGrid((K + TILE_K - 1) / TILE_K,
                       (M + TILE_M - 1) / TILE_M);

    matrix_multiplication_kernel_1D_Register<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    // cudaDeviceSynchronize();
}