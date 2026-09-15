#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <string>

#include "cuda_check.cuh"

struct BenchmarkResult {
    float avg_ms;
};

template <typename Func>
BenchmarkResult benchmark_kernel(
    const std::string& name,
    Func&& func,
    int warmup = 10,
    int iterations = 10)
{
    // =========================
    // 1. Warmup
    // =========================

    for (int i = 0; i < warmup; ++i) {
        func();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // =========================
    // 2. 创建 CUDA Event
    // =========================

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // =========================
    // 3. 正式计时
    // =========================

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        func();
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // =========================
    // 4. 获取总时间
    // =========================

    float total_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );

    float avg_ms = total_ms / iterations;

    // =========================
    // 5. 释放 Event
    // =========================

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // =========================
    // 6. 输出结果
    // =========================

    std::cout << name << ": " << avg_ms << " ms" << std::endl;

    return {avg_ms};
}