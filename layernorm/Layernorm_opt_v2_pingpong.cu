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

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warpReduceSum(val); // Final reduce within the first warp
    }
    __syncthreads();
    return val;
}

// ---------------------------------------------------------------------------------
// 优化: 小数据切分并行 (利用 L2/Global Memory 做跨 Block 规约)
// ---------------------------------------------------------------------------------
__global__ void layernorm_opt_small_part1(const float* __restrict__ x, float* __restrict__ workspace_sum, float* __restrict__ workspace_sq_sum, size_t B, size_t dim) {
    // gridDim.x = blocks_per_row, gridDim.y = B
    size_t row = blockIdx.y;
    if (row >= B) return;

    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;
    size_t bdim = blockDim.x;
    size_t gdim = gridDim.x;

    const float* row_x = x + row * dim;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    bool can_vectorize = ((reinterpret_cast<size_t>(row_x) % 16) == 0);
    size_t dim_aligned = can_vectorize ? (dim / 4) * 4 : 0;

    // 跨 Block 步长
    size_t stride = bdim * gdim * 4;
    size_t start = (bid * bdim + tid) * 4;

    float4 curr_val;
    if (start < dim_aligned) {
        curr_val = *reinterpret_cast<const float4*>(&row_x[start]);
    }

    for (size_t i = start; i < dim_aligned; i += stride) {
        float4 next_val;
        if (i + stride < dim_aligned) {
            next_val = *reinterpret_cast<const float4*>(&row_x[i + stride]);
        }
        
        local_sum += curr_val.x + curr_val.y + curr_val.z + curr_val.w;
        local_sq_sum += curr_val.x * curr_val.x + curr_val.y * curr_val.y + curr_val.z * curr_val.z + curr_val.w * curr_val.w;
        
        curr_val = next_val;
    }

    // 处理余数部分
    size_t rem_stride = bdim * gdim;
    size_t rem_start = dim_aligned + bid * bdim + tid;
    for (size_t i = rem_start; i < dim; i += rem_stride) {
        float val = row_x[i];
        local_sum += val;
        local_sq_sum += val * val;
    }

    // Block 内部规约
    __shared__ float shared_mem[32];
    float sum = blockReduceSum(local_sum, shared_mem);
    float sq_sum = blockReduceSum(local_sq_sum, shared_mem);

    // Block 0 写入 Global Workspace (L2 Cache)
    if (tid == 0) {
        atomicAdd(&workspace_sum[row], sum);
        atomicAdd(&workspace_sq_sum[row], sq_sum);
    }
}

__global__ void layernorm_opt_small_part2(const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ out, const float* __restrict__ workspace_sum, const float* __restrict__ workspace_sq_sum, size_t B, size_t dim) {
    size_t row = blockIdx.y;
    if (row >= B) return;

    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;
    size_t bdim = blockDim.x;
    size_t gdim = gridDim.x;

    const float* row_x = x + row * dim;
    float* row_out = out + row * dim;

    // 读取全局均值与方差
    float sum = workspace_sum[row];
    float sq_sum = workspace_sq_sum[row];

    float mean = sum / dim;
    float var = (sq_sum / dim) - (mean * mean);
    float inv_std = rsqrtf(var + 1e-5f);

    bool can_vectorize = ((reinterpret_cast<size_t>(row_x) % 16) == 0) && ((reinterpret_cast<size_t>(row_out) % 16) == 0) && ((reinterpret_cast<size_t>(gamma) % 16) == 0) && ((reinterpret_cast<size_t>(beta) % 16) == 0);
    size_t dim_aligned = can_vectorize ? (dim / 4) * 4 : 0;

    size_t stride = bdim * gdim * 4;
    size_t start = (bid * bdim + tid) * 4;

    float4 curr_val_out, curr_g, curr_b;
    if (start < dim_aligned) {
        curr_val_out = *reinterpret_cast<const float4*>(&row_x[start]);
        curr_g = *reinterpret_cast<const float4*>(&gamma[start]);
        curr_b = *reinterpret_cast<const float4*>(&beta[start]);
    }

    for (size_t i = start; i < dim_aligned; i += stride) {
        float4 next_val_out, next_g, next_b;
        if (i + stride < dim_aligned) {
            next_val_out = *reinterpret_cast<const float4*>(&row_x[i + stride]);
            next_g = *reinterpret_cast<const float4*>(&gamma[i + stride]);
            next_b = *reinterpret_cast<const float4*>(&beta[i + stride]);
        }
        
        float4 out_val;
        out_val.x = (curr_val_out.x - mean) * inv_std * curr_g.x + curr_b.x;
        out_val.y = (curr_val_out.y - mean) * inv_std * curr_g.y + curr_b.y;
        out_val.z = (curr_val_out.z - mean) * inv_std * curr_g.z + curr_b.z;
        out_val.w = (curr_val_out.w - mean) * inv_std * curr_g.w + curr_b.w;
        
        *reinterpret_cast<float4*>(&row_out[i]) = out_val;

        curr_val_out = next_val_out;
        curr_g = next_g;
        curr_b = next_b;
    }

    size_t rem_stride = bdim * gdim;
    size_t rem_start = dim_aligned + bid * bdim + tid;
    for (size_t i = rem_start; i < dim; i += rem_stride) {
        row_out[i] = (row_x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
// ---------------------------------------------------------------------------------

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

    // 1. 向量化访存 (float4) 与循环展开 (Ping-pong buffering)
    // 将数据按 float4 处理，提高内存带宽利用率
    // 检查指针是否 16 字节对齐
    bool can_vectorize = ((reinterpret_cast<size_t>(row_x) % 16) == 0) && ((reinterpret_cast<size_t>(row_out) % 16) == 0) && ((reinterpret_cast<size_t>(gamma) % 16) == 0) && ((reinterpret_cast<size_t>(beta) % 16) == 0);
    size_t dim_aligned = can_vectorize ? (dim / 4) * 4 : 0;
    
    // Ping-pong buffering for load
    float4 curr_val;
    if (tid * 4 < dim_aligned) {
        curr_val = *reinterpret_cast<const float4*>(&row_x[tid * 4]);
    }

    for (size_t i = tid * 4; i < dim_aligned; i += blockDim.x * 4) {
        float4 next_val;
        if (i + blockDim.x * 4 < dim_aligned) {
            next_val = *reinterpret_cast<const float4*>(&row_x[i + blockDim.x * 4]);
        }
        
        local_sum += curr_val.x + curr_val.y + curr_val.z + curr_val.w;
        local_sq_sum += curr_val.x * curr_val.x + curr_val.y * curr_val.y + curr_val.z * curr_val.z + curr_val.w * curr_val.w;
        
        curr_val = next_val;
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

    // 3. 结果写回 (Ping-pong buffering)
    float4 curr_val_out, curr_g, curr_b;
    if (tid * 4 < dim_aligned) {
        curr_val_out = *reinterpret_cast<const float4*>(&row_x[tid * 4]);
        curr_g = *reinterpret_cast<const float4*>(&gamma[tid * 4]);
        curr_b = *reinterpret_cast<const float4*>(&beta[tid * 4]);
    }

    for (size_t i = tid * 4; i < dim_aligned; i += blockDim.x * 4) {
        float4 next_val_out, next_g, next_b;
        if (i + blockDim.x * 4 < dim_aligned) {
            next_val_out = *reinterpret_cast<const float4*>(&row_x[i + blockDim.x * 4]);
            next_g = *reinterpret_cast<const float4*>(&gamma[i + blockDim.x * 4]);
            next_b = *reinterpret_cast<const float4*>(&beta[i + blockDim.x * 4]);
        }
        
        float4 out_val;
        out_val.x = (curr_val_out.x - mean) * inv_std * curr_g.x + curr_b.x;
        out_val.y = (curr_val_out.y - mean) * inv_std * curr_g.y + curr_b.y;
        out_val.z = (curr_val_out.z - mean) * inv_std * curr_g.z + curr_b.z;
        out_val.w = (curr_val_out.w - mean) * inv_std * curr_g.w + curr_b.w;
        
        *reinterpret_cast<float4*>(&row_out[i]) = out_val;

        curr_val_out = next_val_out;
        curr_g = next_g;
        curr_b = next_b;
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

    // 启动优化后的 kernel: 根据数据量选择策略
    if (B < 768) {
        // (2) 小数据切分并行：M较小时，利用多个Block处理一行，避免SM空闲
        // 动态计算线程数和跨行Block数
        int total_threads_needed = (dim + 3) / 4;
        int threads_per_block = min(256, ((total_threads_needed + 31) / 32) * 32);
        int blocks_per_row = (total_threads_needed + threads_per_block - 1) / threads_per_block;
        
        // 分配 Global Workspace 并初始化为0
        float* workspace_sum_d;
        float* workspace_sq_sum_d;
        cudaMalloc((void**)&workspace_sum_d, B * sizeof(float));
        cudaMalloc((void**)&workspace_sq_sum_d, B * sizeof(float));
        cudaMemset(workspace_sum_d, 0, B * sizeof(float));
        cudaMemset(workspace_sq_sum_d, 0, B * sizeof(float));

        dim3 grid(blocks_per_row, B);
        
        layernorm_opt_small_part1<<<grid, threads_per_block>>>(x_d, workspace_sum_d, workspace_sq_sum_d, B, dim);
        cudaDeviceSynchronize(); // 确保 Part1 完成
        layernorm_opt_small_part2<<<grid, threads_per_block>>>(x_d, gamma_d, beta_d, out_d, workspace_sum_d, workspace_sq_sum_d, B, dim);
        
        cudaFree(workspace_sum_d);
        cudaFree(workspace_sq_sum_d);
    } else {
        // M 较大时，每个 Block 处理一行，但仍需根据 N 动态调整线程数以避免浪费
        int total_threads_needed = (dim + 3) / 4;
        int threads_per_block = min(256, ((total_threads_needed + 31) / 32) * 32);
        if (threads_per_block == 0) threads_per_block = 32;

        int blocks_per_grid = B;
        layernorm_opt<<<blocks_per_grid, threads_per_block>>>(x_d, gamma_d, beta_d, out_d, B, dim);
    }

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
