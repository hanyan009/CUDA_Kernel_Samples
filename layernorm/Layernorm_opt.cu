#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>

using namespace std;

// Warp-level reduction
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset); // 自带同步功能。执行之前，所有线程都要运行到这一行。
    }
    return val;
}

// Block-level reduction
__inline__ __device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val); // Each warp reduces its own sum

    if (lane == 0) {
        shared[wid] = val; // Write warp sum to shared memory
    }
    __syncthreads(); // Wait for all warps

    // Read from shared memory only if that warp existed
    // 【问】threadIdx.x < blockDim.x / warpSize 是什么？
    // 【答】blockDim.x 是总线程数、warpSize 是恒定32。
    // 【重要结论】blockDim.x / warpSize 是总warp数。
    // 【结果】所以这是遍历所有的warp，对warp再做一次归约
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warpReduceSum(val); // Final reduce within the first warp
    }
    __syncthreads();
    return val;
}

// 优化的 LayerNorm Kernel (基于 SIMT 架构的 Block 级并行 + 向量化访存 + 寄存器规约)
__global__ void layernorm_opt(const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ out, size_t B, size_t dim) {
    // 每一个 Block 处理一行（一个 batch）
    size_t row = blockIdx.x;
    if (row >= B) return;

    size_t tid = threadIdx.x;
    
    // 指向当前行的数据起点
    const float* row_x = x + row * dim;
    float* row_out = out + row * dim;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    // 1. 向量化访存 (float4) 与循环展开
    // 将数据按 float4 处理，提高内存带宽利用率
    // 检查指针是否 16 字节对齐
    bool can_vectorize = ((reinterpret_cast<size_t>(row_x) % 16) == 0) && ((reinterpret_cast<size_t>(row_out) % 16) == 0) && ((reinterpret_cast<size_t>(gamma) % 16) == 0) && ((reinterpret_cast<size_t>(beta) % 16) == 0);
    size_t dim_aligned = can_vectorize ? (dim / 4) * 4 : 0;
    
    for (size_t i = tid * 4; i < dim_aligned; i += blockDim.x * 4) {
        float4 val = *reinterpret_cast<const float4*>(&row_x[i]);
        
        local_sum += val.x + val.y + val.z + val.w;
        local_sq_sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    // 处理余数部分 (Remainder handling)
    for (size_t i = dim_aligned + tid; i < dim; i += blockDim.x) {
        float val = row_x[i];
        local_sum += val;
        local_sq_sum += val * val;
    }

    // 2. Block 级别规约 (利用 Warp Shuffle 和 Shared Memory)
    __shared__ float shared_mem[32]; // 为什么共享内存建立为 32？这用来存每个warp的归约值，那么能处理的最大的数量就是32*32=1024，这是block内线程数的最大值。
    float sum = blockReduceSum(local_sum, shared_mem);
    float sq_sum = blockReduceSum(local_sq_sum, shared_mem);

    // 计算均值和方差，由 thread 0 计算后广播给所有 thread
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    if (tid == 0) {
        s_mean = sum / dim;
        float var = (sq_sum / dim) - (s_mean * s_mean);
        s_inv_std = rsqrtf(var + 1e-5f);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    // 3. 结果写回 (同样使用 float4 向量化)
    for (size_t i = tid * 4; i < dim_aligned; i += blockDim.x * 4) {
        float4 val = *reinterpret_cast<const float4*>(&row_x[i]);
        float4 g = *reinterpret_cast<const float4*>(&gamma[i]);
        float4 b = *reinterpret_cast<const float4*>(&beta[i]);
        
        float4 out_val;
        out_val.x = (val.x - mean) * inv_std * g.x + b.x;
        out_val.y = (val.y - mean) * inv_std * g.y + b.y;
        out_val.z = (val.z - mean) * inv_std * g.z + b.z;
        out_val.w = (val.w - mean) * inv_std * g.w + b.w;
        
        *reinterpret_cast<float4*>(&row_out[i]) = out_val;
    }

    // 写回余数部分
    for (size_t i = dim_aligned + tid; i < dim; i += blockDim.x) {
        row_out[i] = (row_x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

__host__ int main() {
    cout << "test Started" << endl;
    constexpr int B = 2;
    // 使用大一点的 dim 才能体现向量化的作用，需为 4 的倍数或者包含余数测试
    constexpr int dim = 5; 
    constexpr int N = B * dim;

    float* x_h = (float *)malloc(N * sizeof(float));
    float* out_h = (float *)malloc(N * sizeof(float));
    float* gamma_h = (float *)malloc(dim * sizeof(float));
    float* beta_h = (float *)malloc(dim * sizeof(float));

    for(int i = 0; i < N; i++){
        x_h[i] = i;
        out_h[i] = 0;
    }
    for(int i = 0; i < dim; i++){
        gamma_h[i] = 1.0f; // 缩放系数初始化为1
        beta_h[i] = 0.0f;  // 偏移系数初始化为0.5测试效果
    }

    float* x_d = nullptr;
    float* out_d = nullptr;
    float* gamma_d = nullptr;
    float* beta_d = nullptr;
    cudaMalloc((void**)&x_d, N * sizeof(float));
    cudaMalloc((void**)&out_d, N * sizeof(float));
    cudaMalloc((void**)&gamma_d, dim * sizeof(float));
    cudaMalloc((void**)&beta_d, dim * sizeof(float));
    
    cudaMemcpy(x_d, x_h, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma_d, gamma_h, dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_d, beta_h, dim * sizeof(float), cudaMemcpyHostToDevice);

    for(size_t i = 0; i < N ; i++){
        if (i % dim == 0 && i != 0) cout << endl;
        cout << x_h[i] << " ";
    }
    cout << "\n<==x on host\n" << endl;

    // 启动优化后的 kernel: 每个 Block 处理一个 Batch(行)
    int threads_per_block = 256;
    int blocks_per_grid = B;
    layernorm_opt<<<blocks_per_grid, threads_per_block>>>(x_d, gamma_d, beta_d, out_d, B, dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "Kernel launch error: " << cudaGetErrorString(err) << endl;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cout << "Kernel execution error: " << cudaGetErrorString(err) << endl;
    }

    cudaMemcpy(out_h, out_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < N ; i++){
        if (i % dim == 0 && i != 0) cout << endl;
        cout << out_h[i] << " ";
    }
    cout << "\n<==out on host" << endl;
    
    free(x_h);
    free(out_h);
    free(gamma_h);
    free(beta_h);
    cudaFree(x_d);
    cudaFree(out_d);
    cudaFree(gamma_d);
    cudaFree(beta_d);
    return 0;
}
