#include <cuda_runtime.h>

#include <iostream>

#include "../common/benchmark.cuh"
#include "../common/cuda_check.cuh"

#include "kernels.cuh"

int main()
{
    // =========================
    // 基本参数
    // =========================

    const int M = 8192;
    const int N = 6144;
    const int K = 4096;
    size_t size_A = static_cast<size_t>(M) * N;
    size_t size_B = static_cast<size_t>(N) * K;
    size_t size_C = static_cast<size_t>(M) * K;

    // =========================
    // Host Memory
    // =========================

    float* h_A = new float[size_A];
    float* h_B = new float[size_B];
    float* h_C = new float[size_C];

    for (size_t i = 0; i < size_A; ++i) {
        h_A[i] = 1.0f;
    }

    for (size_t i = 0; i < size_B; ++i) {
        h_B[i] = 1.0f;
    }

    // =========================
    // GPU Memory
    // =========================

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A,size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B,size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C,size_C * sizeof(float)));

    // =========================
    // Host -> Device
    // =========================

    CUDA_CHECK(cudaMemcpy(d_A,h_A,size_A * sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B,h_B,size_B * sizeof(float),cudaMemcpyHostToDevice));

    // =========================
    // Benchmark naive
    // =========================

    auto result_naive = benchmark_kernel("matrix_multiplication_naive",
        [&]() {solve_naive(d_A, d_B, d_C, M, N, K);},5,5);

    // =========================
    // Benchmark tiled
    // =========================

    auto result_tiled = benchmark_kernel("matrix_multiplication_tiled",
        [&]() {solve_tiled(d_A, d_B, d_C, M, N, K);},5,5);

    // =========================
    // Device -> Host
    // =========================

    CUDA_CHECK(cudaMemcpy(h_C,d_C,size_C * sizeof(float),cudaMemcpyDeviceToHost));

    // =========================
    // 释放资源
    // =========================

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    return 0;
}