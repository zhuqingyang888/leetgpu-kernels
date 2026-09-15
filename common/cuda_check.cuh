#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

inline void cuda_check(
    cudaError_t error,
    const char* file,
    int line)
{
    if (error != cudaSuccess) {
        std::cerr
            << "CUDA Error: "
            << cudaGetErrorString(error)
            << "\nFile: " << file
            << "\nLine: " << line
            << std::endl;

        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) \
    cuda_check((call), __FILE__, __LINE__)