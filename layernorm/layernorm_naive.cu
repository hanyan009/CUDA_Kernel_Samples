#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
using namespace std;


// __global__ 

__host__ int main(){
    // 测试用例
    cout << "test" << endl;
    constexpr int dim = 10;
    float* x_h = (float *)malloc(dim * sizeof(float));
    float* out_h = (float *)malloc(dim * sizeof(float));

    for(int i = 0; i < dim; i++){
        x_h[i] = i;
        out_h[i] = 0;
    }

    float* x_d = nullptr;
    float* out_d = nullptr;
    cudaMalloc((void**)&x_d, dim * sizeof(float)); // 因为要修改这个指针，要传入指针的地址（二重指针）。
    cudaMalloc((void**)&out_d, dim * sizeof(float));
    cudaMemcpy(x_d, x_h, dim * sizeof(float), cudaMemcpyHostToDevice); // 【记忆】先写target，再写源数据，然后是方向。写方向是为了方便管理。
    cudaMemcpy(out_d, out_h, dim * sizeof(float), cudaMemcpyHostToDevice);
    

}