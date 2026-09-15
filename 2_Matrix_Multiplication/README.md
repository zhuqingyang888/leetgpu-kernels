## 1. V0：Naive Matrix Multiplication

最基础的 CUDA 矩阵乘法实现，每个 CUDA 线程负责计算输出矩阵 `C` 中的一个元素：

```cpp
float sum = 0.0f;

for (int i = 0; i < N; ++i) {
    sum += A[row * N + i] * B[i * K + col];
}

C[row * K + col] = sum;
```

初始线程块配置：

```cpp
dim3 threadsPerBlock(16, 16);
```

### Block Size 测试

在不修改 Kernel 计算逻辑的情况下，对不同线程块形状进行测试：

| Block Size | Threads / Block | Time |
|---|---:|---:|
| 16 × 16 | 256 | 419 ms |
| 32 × 8 | 256 | 509 ms |
| 8 × 32 | 256 | 418 ms |
| 32 × 32 | 1024 | 474 ms |

实验表明，单纯增大线程块或者令 `blockDim.x = 32` 并不能保证获得更好的性能。

其中 `32 × 8` 与 `8 × 32` 虽然线程总数相同，但 warp 在线程块中的二维映射不同，因此 A、B 的访存模式也不同。

本阶段最终继续采用：

```cpp
dim3 threadsPerBlock(16, 16);
```

作为后续优化的基础配置。

---

## 2. V2：Shared Memory Tiling

Naive Kernel 中，不同线程在矩阵乘法过程中会重复使用大量 A、B 数据。

因此引入 Shared Memory Tiling，将 A、B 的局部 Tile 从 Global Memory 加载到 Shared Memory：

```text
Global Memory
      |
      v
Shared Memory
      |
      v
   Reuse Data
      |
      v
     FMA
```

当前使用：

```cpp
#define TILE_SIZE 16
```

每个 Block 负责计算 C 的一个 `16 × 16` Tile。

核心计算：

```cpp
for (int tile = 0; tile < numTiles; ++tile) {

    sharedA[threadIdx.y][threadIdx.x] = ...;
    sharedB[threadIdx.y][threadIdx.x] = ...;

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; ++i) {
        sum += sharedA[threadIdx.y][i]
             * sharedB[i][threadIdx.x];
    }

    __syncthreads();
}
```

Shared Memory 使一个 Block 内加载的 A、B 数据能够被多个线程重复使用，从而减少对 Global Memory 的重复访问。

### Benchmark

| Version | Time | Speedup |
|---|---:|---:|
| V0 Naive | 417.547 ms | 1.00× |
| V2 Shared Memory Tiled | 296.497 ms | **1.41×** |

相比 Naive Kernel：

- 执行时间降低约 **29%**
- 获得约 **1.41×** 加速

说明 Shared Memory Tiling 对当前矩阵乘法具有明显优化效果。

---

## 3. 当前优化路线

```text
V0  Naive GEMM
      |
      | Block Shape Benchmark
      v
V1  Thread Block 调整
      |
      | 无明显性能提升
      v
V2  Shared Memory Tiling
      |
      | 417.5 ms -> 296.5 ms
      | 1.41x Speedup
      v
V3  1D Register Blocking       <- Next
      |
      v
V4  2D Register Blocking
      |
      v
V5  Vectorized Load (float4)
      |
      v
V6  Double Buffering
      |
      v
V7  Async Copy / Pipeline
      |
      v
V8  Tensor Core / MMA
      |
      v
cuBLAS / CUTLASS Comparison
```

## 4. Next Step

下一步实现 **V3：1D Register Blocking**。

V2 中一个线程只负责计算一个 C 元素：

```text
Thread
  |
  v
1 x 1 output
```

V3 将尝试让一个线程计算多个 C 元素，例如：

```text
Thread
  |
  v
4 x 1 outputs
```

使用多个寄存器保存累加结果：

```cpp
float sum0 = 0.0f;
float sum1 = 0.0f;
float sum2 = 0.0f;
float sum3 = 0.0f;
```

并复用从 Shared Memory 读取的数据：

```cpp
float b = sharedB[i][tx];

sum0 += sharedA[...] * b;
sum1 += sharedA[...] * b;
sum2 += sharedA[...] * b;
sum3 += sharedA[...] * b;
```

目标是进一步提高数据复用率和计算密度，并继续与 V0、V2 进行 Benchmark 对比。

## Progress

- [x] V0 - Naive GEMM
- [x] V1 - Block Shape Benchmark
- [x] V2 - Shared Memory Tiling
- [ ] V3 - 1D Register Blocking
- [ ] V4 - 2D Register Blocking
- [ ] V5 - Vectorized Memory Access
- [ ] V6 - Double Buffering
- [ ] V7 - Async Pipeline
- [ ] V8 - Tensor Core
- [ ] cuBLAS / CUTLASS Comparison