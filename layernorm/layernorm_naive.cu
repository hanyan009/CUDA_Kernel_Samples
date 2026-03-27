#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
using namespace std;


// 【注】朴素
__global__ void layernorm(float* x, float* out, float* gamma, float* beta, size_t B, size_t dim){
    // x[] = ;
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= B) return; // 边界保护

    float sum = 0;
    float square_sum = 0;
    // 单次循环获取方差和均值。
    // 1 累计求和
    // 2 平方和
    for(size_t i = 0; i < dim; i++){
        float val = x[(thread_idx * dim) + i];
        sum += val / dim;
        square_sum += val * val / dim;
    }
    // float mu = ;
    float var = square_sum - sum*sum;
    float inv_std = rsqrtf(var + 1e-5f); // 【重要】rsqrtf求开根再计算

    for(size_t i = 0; i < dim; i++){
        float normalized = (x[(thread_idx * dim) + i] - sum) * inv_std;
        out[(thread_idx * dim) + i] = normalized * gamma[i] + beta[i];
    }

}

__host__ int main(){
    // 测试用例
    cout << "test Started" << endl;
    constexpr int B = 2;
    constexpr int dim = 5;
    constexpr int N = B * dim;

    float* x_h = (float *)malloc( N * sizeof(float));
    float* out_h = (float *)malloc( N * sizeof(float));
    float* gamma_h = (float *)malloc( dim * sizeof(float));
    float* beta_h = (float *)malloc( dim * sizeof(float));

    for(int i = 0; i <  N; i++){
        x_h[i] = i;
        out_h[i] = 0;
    }
    for(int i = 0; i < dim; i++){
        gamma_h[i] = 1.0f; // 缩放因子初始化为1
        beta_h[i] = 0.0f;  // 偏置初始化为0
    }

    float* x_d = nullptr;
    float* out_d = nullptr;
    float* gamma_d = nullptr;
    float* beta_d = nullptr;
    cudaMalloc((void**)&x_d,  N * sizeof(float)); // 因为要修改这个指针，要传入指针的地址（二重指针）。
    cudaMalloc((void**)&out_d,  N * sizeof(float));
    cudaMalloc((void**)&gamma_d, dim * sizeof(float));
    cudaMalloc((void**)&beta_d, dim * sizeof(float));
    cudaMemcpy(x_d, x_h,  N * sizeof(float), cudaMemcpyHostToDevice); // 【记忆】先写target，再写源数据，然后是方向。写方向是为了方便管理。
    cudaMemcpy(out_d, out_h,  N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gamma_d, gamma_h, dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_d, beta_h, dim * sizeof(float), cudaMemcpyHostToDevice);

    for(size_t i = 0; i < N ; i++){
        if (i % dim == 0) cout << endl;
        cout << x_h[i] << " ";
    }
    cout << "<==x on host" << endl;

    // 启动kernel（调用global函数）
    if(B < 256) layernorm<<<B, 256>>>(x_d, out_d, gamma_d, beta_d, B, dim);

    cudaMemcpy(out_h, out_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i <  N ; i++){
        if (i % dim == 0) cout << endl;
        cout << out_h[i] << " ";
    }
    cout << "<==out on host" << endl;
    return 0;
}

