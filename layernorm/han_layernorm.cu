#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
using namespace std;


__inline__ __device__ float warpReduceSum(float val){
    // 这里循环的是折半，起始是32/2=16，然后每次折半
    for(size_t offset=warpSize / 2; offset > 0 ; offset = offset / 2){
        // 【__shfl_down_sync有同步含义】所有线程准备完成数据后，进行取数据
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val, float* shm_val){
    size_t warpId = threadIdx.x / warpSize;
    size_t lane = threadIdx.x % warpSize;

    float sum = warpReduceSum(val);
    if (lane == 0) {
        shm_val[warpId] = sum; // 只有每个 warp 的首线程写入共享内存，避免竞态
    }

    __syncthreads(); // 等待所有 warp 归约完毕

    float result = 0.0f;
    if (warpId == 0) { // 第一个warp再将 shared memo 中的数据进行一次归约
        // 确保不越界读取，并且只累加有效 warp 的结果（未初始化的位置补0）
        float temp = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shm_val[lane] : 0.0f;
        result = warpReduceSum(temp);
    }
    __syncthreads(); // 等待第一个 warp 汇总完成，确保共享内存可以被安全复用
    return result;
}

__global__ void layernorm_han(const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ out, size_t B, size_t dim) {
    // 第一步取出当前线程要操作的batch的指针。
    size_t row = blockIdx.x; // 第几个向量
    const float* row_x = x + row * dim; // 取到输入指针
    float* row_out = out + row * dim; // 取到输出指针

    // 判断是否可float4向量化。也就是判断字节是否都对齐。不对齐会很慢
    // 修复：这里必须判断当前行的指针 row_x 和 row_out，而不是总指针 x 和 out
    bool can_vectorize = (reinterpret_cast<size_t>(row_x) % 16 == 0) && 
                         (reinterpret_cast<size_t>(gamma) % 16 == 0) && 
                         (reinterpret_cast<size_t>(beta) % 16 == 0) && 
                         (reinterpret_cast<size_t>(row_out) % 16 == 0);
                         
    // 可对齐。那么可能会有余数，这里取前面对齐的进行float4向量化，再取余数
    size_t dim_aligned = can_vectorize ? (dim / 4) * 4 : 0;

    //累加变量
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    // 【float4】向量化的网格跨步循环
    for (size_t i = threadIdx.x * 4; i < dim_aligned; i += blockDim.x * 4) {
        float4 x4 = *reinterpret_cast<const float4*>(&row_x[i]); // 重新 cast 为也带 const 的，靠谱一些
        sum += x4.x + x4.y + x4.z + x4.w;
        sq_sum += x4.x * x4.x + x4.y * x4.y + x4.z * x4.z + x4.w * x4.w;
    }

    // 处理余数：
    // 【起始i加上dim_aligned即可】把剩下的当做一个序列，只改变起始i即可。
    for (size_t i = dim_aligned + threadIdx.x; i < dim; i += blockDim.x) {
        float val = row_x[i]; // 先读到局部变量（寄存器），会更快
        sum += val;
        sq_sum += val * val;
    }

    // 进入warp 归约。经过上面的处理，最长也就是256或者1024个线程上各有一个数。
    // 【block归约，内涵warp归约】现在要把每个线程上的数字，进行归约求和。
    __shared__ float shm_val[32]; // 【问】为什么32？【答】因为线程数最高1024，有1024/32=32个warp。
    float sum_result = blockReduceSum(sum, shm_val);  // 先求出每个 warp 的sum，放入shm_val，再让第一个warp 读这32个，再进行一次归约。
    float sqsum_result = blockReduceSum(sq_sum, shm_val);

    __shared__ float mean;
    __shared__ float inv_std;
    // 计算均值、方差
    if (threadIdx.x == 0) { // 第0个线程计算即可
        mean = sum_result / dim;
        float var = sqsum_result / dim - mean * mean;
        inv_std = rsqrtf(var + 1e-5f);
    }
    __syncthreads(); // 等待第0线程计算完毕均值和方差

    // 按元素处理：归一化与仿射变换
    for (size_t i = threadIdx.x * 4; i < dim_aligned; i += blockDim.x * 4) {
        float4 x4 = *reinterpret_cast<const float4*>(&row_x[i]);
        float4 g = *reinterpret_cast<const float4*>(&gamma[i]);
        float4 b = *reinterpret_cast<const float4*>(&beta[i]);

        float4 out4; // 先写入局部变量（寄存器），再一次性写回输出的变量，效率更高
        out4.x = (x4.x - mean) * inv_std * g.x + b.x; 
        out4.y = (x4.y - mean) * inv_std * g.y + b.y; 
        out4.z = (x4.z - mean) * inv_std * g.z + b.z; 
        out4.w = (x4.w - mean) * inv_std * g.w + b.w; 

        *reinterpret_cast<float4*>(&row_out[i]) = out4;
    }
    // 处理余数：
    for (size_t i = dim_aligned + threadIdx.x; i < dim; i += blockDim.x) {
        float val = row_x[i]; // 先读到局部变量（寄存器），会更快
        float g = gamma[i];
        float b = beta[i];

        row_out[i] = (val - mean) * inv_std * g + b;
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
        beta_h[i] = 0.0f;  // 偏移系数初始化为0测试效果
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
    layernorm_han<<<blocks_per_grid, threads_per_block>>>(x_d, gamma_d, beta_d, out_d, B, dim);

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

