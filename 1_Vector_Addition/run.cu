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
    const int N = (2 << 27) + 3;//向量长度

    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    // =========================
    // Host Memory
    // =========================

    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // =========================
    // Device Memory
    // =========================

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // =========================
    // Benchmark Naive
    // =========================
    auto result_naive =
        benchmark_kernel(
            "vector_add_naive",
            [&]() {solve_naive(d_A, d_B, d_C, N);}
        );

    CUDA_CHECK(cudaGetLastError());

    // =========================
    // Benchmark grid stride
    // =========================
    auto result_grid_stride =
        benchmark_kernel(
            "vector_add_grid_stride",
            [&]() {solve_grid_stride(d_A, d_B, d_C, N);}
        );

    CUDA_CHECK(cudaGetLastError());

    // =========================
    // Benchmark float4
    // =========================
    auto result_float4 =
        benchmark_kernel(
            "vector_add_float4",
            [&]() {solve_float4(d_A, d_B, d_C, N);}
        );

    auto result_float4_V1 =
        benchmark_kernel(
            "vector_add_float4_V1",
            [&]() {solve_float4_V1(d_A, d_B, d_C, N);}
        );

    CUDA_CHECK(cudaGetLastError());

    auto result_float4_x2 =
        benchmark_kernel(
            "vector_add_float4_x2",
            [&]() {solve_float4_x2(d_A, d_B, d_C, N);}
        );

    CUDA_CHECK(cudaGetLastError());

    // =========================
    // 结果拷回 CPU
    // =========================

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));


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