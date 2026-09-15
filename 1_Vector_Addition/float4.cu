// N不是4的倍数单独处理
__global__ void vector_add_float4(const float4* __restrict__ A, const float4* __restrict__ B, float* __restrict__ C, int N){    
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if(idx < N / 4){
        reinterpret_cast<float4*>(C)[idx] = make_float4(A[idx].x + B[idx].x, A[idx].y + B[idx].y,
            A[idx].z + B[idx].z, A[idx].w + B[idx].w);
    }
    else if(idx = N / 4){
        for(int i = N - N % 4; i < N; i++){
            C[idx] = reinterpret_cast<const float*>(A)[idx] + reinterpret_cast<const float*>(B)[idx];
        }
    }
}


extern "C" void solve_float4(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = ((N + 3) / 4 + threadsPerBlock - 1) / threadsPerBlock;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);

    vector_add_float4<<<blocksPerGrid, threadsPerBlock>>>(A4, B4, C, N);
    // cudaDeviceSynchronize();
}



// N不是4的倍数，串行处理尾部
__global__ void vector_add_float4_V1(const float4* __restrict__ A, const float4* __restrict__ B, float4* __restrict__ C, int N){    
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if(idx < N){
        reinterpret_cast<float4*>(C)[idx] = make_float4(A[idx].x + B[idx].x, A[idx].y + B[idx].y,
            A[idx].z + B[idx].z, A[idx].w + B[idx].w);
    }
}

__global__ void vector_add_tail(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int tail ,int start){
    int idx = threadIdx.x + start;
    if(threadIdx.x < tail){
        C[idx] = A[idx] + B[idx];
    }
}


extern "C" void solve_float4_V1(const float* A, const float* B, float* C, int N) {
    int N4 = N / 4;
    int tail = N % 4;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N4 + threadsPerBlock - 1) / threadsPerBlock;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    if(N4 > 0){
        vector_add_float4_V1<<<blocksPerGrid, threadsPerBlock>>>(A4, B4, C4, N4);
    }

    if(tail > 0){
        int start = N - tail;
        vector_add_tail<<<1,32>>>(A, B, C, tail, start);
    }
    
    // cudaDeviceSynchronize();
}

// 每个线程处理2个float4
__global__ void vector_add_float4_x2(const float4* __restrict__ A, const float4* __restrict__ B, float4* __restrict__ C, int N8){    
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2; 
    if(idx + 1 < N8 * 2){
        reinterpret_cast<float4*>(C)[idx] = make_float4(A[idx].x + B[idx].x, A[idx].y + B[idx].y,
            A[idx].z + B[idx].z, A[idx].w + B[idx].w);
        reinterpret_cast<float4*>(C)[idx + 1] = make_float4(A[idx + 1].x + B[idx + 1].x, A[idx + 1].y + B[idx + 1].y,
            A[idx + 1].z + B[idx + 1].z, A[idx + 1].w + B[idx + 1].w);
    }
}

extern "C" void solve_float4_x2(const float* A, const float* B, float* C, int N) {
    int N8 = N / 8;
    int tail = N % 8;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N8 + threadsPerBlock - 1) / threadsPerBlock;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    if(N8 > 0){
        vector_add_float4_x2<<<blocksPerGrid, threadsPerBlock>>>(A4, B4, C4, N8);
    }

    if(tail > 0){
        int start = N - tail;
        vector_add_tail<<<1,32>>>(A, B, C, tail, start);
    }
    
    // cudaDeviceSynchronize();
}