#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
using namespace std;


// 【注】朴素
__global__ void layernorm(float* x, float* out, size_t B, size_t dim){
    // x[] = ;
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
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
        out[(thread_idx * dim) + i] = (x[(thread_idx * dim) + i] - sum) * inv_std;
    }

}

__host__ int main(){
    // 测试用例
    cout << "test" << endl;
    constexpr int B = 2;
    constexpr int dim = 5;
    constexpr int N = B * dim;

    float* x_h = (float *)malloc( N * sizeof(float));
    float* out_h = (float *)malloc( N * sizeof(float));

    for(int i = 0; i <  N; i++){
        x_h[i] = i;
        out_h[i] = 0;
    }

    float* x_d = nullptr;
    float* out_d = nullptr;
    cudaMalloc((void**)&x_d,  N * sizeof(float)); // 因为要修改这个指针，要传入指针的地址（二重指针）。
    cudaMalloc((void**)&out_d,  N * sizeof(float));
    cudaMemcpy(x_d, x_h,  N * sizeof(float), cudaMemcpyHostToDevice); // 【记忆】先写target，再写源数据，然后是方向。写方向是为了方便管理。
    cudaMemcpy(out_d, out_h,  N * sizeof(float), cudaMemcpyHostToDevice);

    for(size_t i = 0; i < N ; i++){
        if (i % dim == 0) cout << endl;
        cout << x_h[i] << " ";
    }
    cout << "<==x on host" << endl;

    // 启动kernel（调用global函数）
    if(B < 256) layernorm<<<1, B>>>(x_d, out_d,  B, dim);

    cudaMemcpy(out_h, out_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i <  N ; i++){
        if (i % dim == 0) cout << endl;
        cout << out_h[i] << " ";
    }
    cout << "<==out on host" << endl;
    return 0;
}

