#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// 声明外部核函数 (注意 extern "C" 必须有)
extern "C" __global__ void baseline_cp_async_kernel(float* gmem_in, float* gmem_out, int iterations);
extern "C" __global__ void hopper_pdl_tma_kernel(float* gmem_in, float* gmem_out, int iterations);

int main(int argc, char** argv) {
    int iterations = 100;
    int block_size = 256;
    size_t n_elements = iterations * block_size;
    size_t size = n_elements * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // 确定运行模式
    std::string mode = (argc > 1) ? argv[1] : "base";

    dim3 grid(1);
    dim3 block(block_size);
    
    // 关键修改：直接手动加 256 字节的缓冲，避免在 Host 代码里引用 cuda::pipeline 类型
    // 这样 main.cu 就不需要包含 <cuda/pipeline> 了，彻底解决报错
    size_t smem_size = block_size * 2 * sizeof(float) + 256; 

    if (mode == "optimized") {
        std::cout << "Running Hopper Optimized Kernel..." << std::endl;
        hopper_pdl_tma_kernel<<<grid, block, smem_size>>>(d_in, d_out, iterations);
    } else {
        std::cout << "Running Baseline Kernel..." << std::endl;
        baseline_cp_async_kernel<<<grid, block, smem_size>>>(d_in, d_out, iterations);
    }
    
    cudaDeviceSynchronize();
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}