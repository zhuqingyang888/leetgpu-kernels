# naive.cu
- 普通的向量加法
# grid_stride.cu
- 使用 Grid-Stride Loop，让每个线程循环处理多个元素
# float4.cu
- 使用 float4 向量化访存
- 其中float4与float4V1区别只是不足4的tail处理
- float4x2，一个线程处理两个float4