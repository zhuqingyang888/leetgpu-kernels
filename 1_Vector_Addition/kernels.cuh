#pragma once

#include <cuda_runtime.h>

// Naive Vector Addition
__global__ void vector_add_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N);

extern "C" void solve_naive(const float* A, const float* B, float* C, int N);

// grid stride Vector Addition
__global__ void vector_add_grid_stride(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N);

extern "C" void solve_grid_stride(const float* A, const float* B, float* C, int N);

// float4 Vector Addition
__global__ void vector_add_float4(const float4* __restrict__ A, const float4* __restrict__ B, float* __restrict__ C, int N);
extern "C" void solve_float4(const float* A, const float* B, float* C, int N);
extern "C" void solve_float4_V1(const float* A, const float* B, float* C, int N);
extern "C" void solve_float4_x2(const float* A, const float* B, float* C, int N);