#pragma once

#include <cuda_runtime.h>

// naive matrix_multiplication
__global__ void matrix_multiplication_kernel_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K);

extern "C" void solve_naive(const float* A, const float* B, float* C, int M, int N, int K);

// tiled matrix_multiplication
__global__ void matrix_multiplication_kernel_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K);

extern "C" void solve_tiled(const float* A, const float* B, float* C, int M, int N, int K);