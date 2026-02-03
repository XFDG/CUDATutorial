#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/pipeline>

// 必须加上 extern "C" 以匹配 main.cu 的声明
extern "C" __global__ void baseline_cp_async_kernel(float* __restrict__ gmem_in, float* __restrict__ gmem_out, int iterations) {
    extern __shared__ float smem[]; 
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // 1. 定义管线对象 (这就是第4个参数需要的对象)
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    for (int i = 0; i < iterations; ++i) {
        // 2. 获取令牌
        pipe.producer_acquire();

        // 3. 发起异步拷贝
        // 关键点：第4个参数必须是 pipe 对象本身，不是 pipe.xxx() 的结果
        cuda::memcpy_async(
            &smem[(i % 2) * block_size + tid], 
            &gmem_in[i * block_size + tid], 
            sizeof(float), 
            pipe  // <--- 请确保这里只写了 pipe
        );

        // 4. 提交
        pipe.producer_commit();
        
        // 5. 等待 (Baseline 模式下我们故意制造停顿)
        pipe.consumer_wait();
        
        // 计算
        float val = smem[(i % 2) * block_size + tid];
        val = val * 2.0f + 1.0f; 
        gmem_out[i * block_size + tid] = val;

        // 释放
        pipe.consumer_release();
    }
}