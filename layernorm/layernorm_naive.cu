#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
using namespace std;


__global__ void layernorm(float* x, int ){
    x;

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



    for(size_t i = 0; i <  N ; i++){
        if (i % dim == 0) cout << endl;
        cout << out_h[i] << " ";
    }
    cout << "<==out on host" << endl;
    return 0;
}

