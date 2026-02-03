#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <string>
#include <bits/stdc++.h>
#include <device_launch_parameters.h>
// 辅助函数：根据计算能力返回每个SM的核心数
int _ConvertSMVer2Cores(int major, int minor) {
    // 定义不同架构下，每个SM包含的CUDA核心数 (FP32)
    typedef struct {
        int SM; // 0xMm (hexidecimal notation), M = major version, m = minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192}, // Kepler Generation (SM 3.0)
        {0x32, 192}, // Kepler Generation (SM 3.2)
        {0x35, 192}, // Kepler Generation (SM 3.5)
        {0x37, 192}, // Kepler Generation (SM 3.7)
        {0x50, 128}, // Maxwell Generation (SM 5.0)
        {0x52, 128}, // Maxwell Generation (SM 5.2)
        {0x53, 128}, // Maxwell Generation (SM 5.3)
        {0x60, 64},  // Pascal Generation (SM 6.0)
        {0x61, 128}, // Pascal Generation (SM 6.1)
        {0x62, 128}, // Pascal Generation (SM 6.2)
        {0x70, 64},  // Volta Generation (SM 7.0)
        {0x72, 64},  // Volta Generation (SM 7.2)
        {0x75, 64},  // Turing Generation (SM 7.5)
        {0x80, 64},  // Ampere Generation (SM 8.0)
        {0x86, 128}, // Ampere Generation (SM 8.6)
        {0x89, 128}, // Ada Lovelace (SM 8.9) rtx 4090等
        {0x90, 128}, // Hopper (SM 9.0) H100
        {-1, -1}
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    // 如果没找到对应的架构，通常返回默认值或最新的已知架构数量
    printf("MapSMtoCores for SM %d.%d is undefined. Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}

int main() {
  int deviceCount = 0;
  // 获取当前机器的GPU数量
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }
  for (int dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    // 初始化当前device的属性获取对象
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    // 显存容量
    printf("  Total amount of global memory:                 %.0f MBytes "
             "(%llu bytes)\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
             (unsigned long long)deviceProp.totalGlobalMem);
    // 时钟频率

       // ✅ 新代码
              int myClockRate = 0;
              // 如果你的代码在一个循环里 (for int dev = 0...)，建议把下面的 0 改成 dev
              cudaDeviceGetAttribute(&myClockRate, cudaDevAttrClockRate, 0); 

                     printf( "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
                            "GHz)\n",
                            myClockRate * 1e-3f, myClockRate * 1e-6f);
       
    //printf( "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
      //  "GHz)\n",
        //deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

        
        
    // L2 cache大小
    printf("  L2 Cache Size:                                 %d bytes\n",
             deviceProp.l2CacheSize);
    // high-frequent used
    // 注释见每个printf内的字符串
    printf("  Total amount of shared memory per block:       %zu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Total shared memory per multiprocessor:        %zu bytes\n",
           deviceProp.sharedMemPerMultiprocessor);
    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock); //寄存器数量
    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Max dimension size of a block size (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);

       int sm_count = deviceProp.multiProcessorCount;
    
    // 2. 获取每个 SM 的核心数
    int cores_per_sm = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
    
    // 3. 计算总核心数
    int total_cores = sm_count * cores_per_sm;

    printf("  SM Count (MultiProcessorCount):   %d\n", sm_count);
    printf("  Compute Capability:               %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Maximum number of cores:          %d\n", total_cores);
  }
  return 0;
}

