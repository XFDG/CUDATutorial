#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

// 必须针对 sm_90 编译
extern "C" __global__ void hopper_pdl_tma_kernel(float* __restrict__ gmem_in, float* __restrict__ gmem_out, int iterations) {
    // 1. 定义类型别名
    using BarrierType = cuda::pipeline_shared_state<cuda::thread_scope_block, 2>;
    
    // 2. 技巧：使用不带构造函数的 raw memory 来避免 "dynamic initialization" 警告
    // 我们利用 extern __shared__ 分配的总空间，从中切出一部分给 pipeline state
    extern __shared__ char smem_buffer[];
    
    // 将共享内存的前几个字节作为 Pipeline 状态
    BarrierType* shared_state = reinterpret_cast<BarrierType*>(smem_buffer);
    
    // 真正的数据缓冲区从 state 之后开始
    float* smem_data = reinterpret_cast<float*>(&smem_buffer[sizeof(BarrierType)]);

    // 初始化 Pipeline (只需在 block 内的一个线程做)
    auto block = cooperative_groups::this_thread_block();
    
    // make_pipeline 会自动处理 shared_state 的初始化
    auto pipe = cuda::make_pipeline(block, shared_state);

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    for (int i = 0; i < iterations; ++i) {
        // 2. 生产者阶段：发起拷贝
        pipe.producer_acquire();
            cuda::memcpy_async(
                &smem_data[(i % 2) * block_size + tid], 
                &gmem_in[i * block_size + tid], 
                sizeof(float), // 直接用普通大小，或者用 cuda::aligned_size_t<4>(sizeof(float))
                pipe
            );
        pipe.producer_commit();
        // 3. 消费者阶段
        pipe.consumer_wait();
        
        // 模拟计算
        float val = smem_data[(i % 2) * block_size + tid];
        gmem_out[i * block_size + tid] = val;

        pipe.consumer_release();
    }
}