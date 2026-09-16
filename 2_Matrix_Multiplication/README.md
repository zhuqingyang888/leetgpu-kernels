# CUDA Matrix Multiplication Optimization

使用 CUDA 手写矩阵乘法 Kernel，并逐步进行性能优化。

矩阵定义：

```text
A: M × N
B: N × K
C: M × K
```

## Environment

```text
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
CUDA: 12.8
Compute Capability: 12.0
SM Count: 26
```

## Optimization

### V0 - Naive

一个线程计算一个 C 元素，直接从 Global Memory 读取 A、B。

```text
Time: 417.793 ms
```

### V1 - Block Shape Tuning

测试不同 Thread Block：

| Block | Time |
|---|---:|
| 16 × 16 | ~418 ms |
| 32 × 8 | ~509 ms |
| 8 × 32 | ~418 ms |
| 32 × 32 | ~474 ms |

仅调整 Block Shape 没有明显提升。

### V2 - Shared Memory Tiling

使用 Shared Memory 缓存 A、B Tile，提高数据复用。

```text
TILE_SIZE = 16

Time: 297.668 ms
Speedup: ~1.40×
```

### V3 - 1D Register Blocking

一个线程沿 M 方向计算多个 C 元素：

```cpp
float sum[TM];
```

增加 Register-Level Data Reuse。

固定：

```cpp
TILE_M = 64
TILE_N = 16
TILE_K = 16
```

测试不同 TM：

| TM | Time |
|---:|---:|
| 1 | 543.349 ms |
| 2 | 283.351 ms |
| 4 | 170.785 ms |
| 8 | 139.596 ms |
| **16** | **130.520 ms** |
| 32 | 256.704 ms |
| 64 | 654.931 ms |

当前配置下 `TM=16` 最优。

同时使用 Cooperative Loading，使 Shared Memory 加载不再依赖固定的 Thread Block Shape。

进一步调整 Tile：

```cpp
TILE_M = 64
TILE_N = 32
TILE_K = 32
TM     = 16
```

得到当前最佳结果：

```text
Time: 98.7917 ms
Speedup vs Naive: ~4.23×
```

## Performance

| Version | Optimization | Time |
|---|---|---:|
| V0 | Naive | 417.793 ms |
| V2 | Shared Memory Tiling | 297.668 ms |
| V3 | 1D Register Blocking | **98.7917 ms** |

## Roadmap

- [x] Naive
- [x] Block Shape Tuning
- [x] Shared Memory Tiling
- [x] 1D Register Blocking
- [ ] 2D Register Blocking
- [ ] Vectorized Load
- [ ] Double Buffering
- [ ] Async Copy / Pipeline
- [ ] Tensor Core
- [ ] cuBLAS / CUTLASS Comparison