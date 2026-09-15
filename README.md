# LeetGPU CUDA 算子刷题记录

记录 LeetGPU 刷题过程中使用 CUDA 编写和优化 GPU 算子的过程。

## 项目目标

- 使用 CUDA 实现 LeetGPU 算子
- 在 WSL + NVIDIA GPU 环境下运行
- 对不同实现进行 Benchmark
- 分析不同优化方法的性能差异
- 记录学习和优化过程

## 实验环境

- 系统：WSL2
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU
- CUDA：12.8
- 编程语言：CUDA C++

## NVIDIA GeForce RTX 5060 Laptop GPU
- Max threads per block: 1024
- blockDim.x <= 1024 blockDim.y <= 1024 blockDim.z <= 64
- Max grid dimensions: 2147483647 x 65535 x 65535
- SM count = 26;
- Max threads per SM = 1536
- Warp size = 32

## Benchmark

统一使用 CUDA Event 对 Kernel 进行计时，并进行多次运行取平均值。
