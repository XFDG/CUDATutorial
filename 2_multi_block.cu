#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 核心改动1：Kernel需要知道数据的三维尺寸(nx, ny, nz)才能计算一维索引
__global__ void sum_3d(float *x, int nx, int ny, int nz)
{
    // 1. 计算当前线程在全局范围内的 3D 坐标
    // 类似于：我是第几栋楼(block)的第几层(thread) -> 绝对楼层
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // Col 列
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // Row 行
    int iz = blockIdx.z * blockDim.z + threadIdx.z; // Depth 深/层

    // 2. 边界检查（非常重要）：防止线程跑出数据范围
    if (ix < nx && iy < ny && iz < nz) {
        
        // 3. 【核心公式】将 3D 坐标映射回 1D 物理内存地址
        // 想象成切方块面包：
        // 先找到是第几片面包 (iz)，跳过前面的所有片 (iz * 一片的面积)
        // 再找到是第几行 (iy)，跳过前面的行 (iy * 一行的宽度)
        // 最后加上列号 (ix)
        int global_tid = iz * (ny * nx) + iy * nx + ix;

        printf("Thread(x,y,z)=[%d,%d,%d] maps to Linear Index=%d\n", ix, iy, iz, global_tid);
        
        x[global_tid] += 1;
    }
}

int main(){
    // 定义三维数据的尺寸
    int nx = 4; // x方向长度
    int ny = 4; // y方向长度
    int nz = 2; // z方向长度
    
    int N = nx * ny * nz; // 总元素个数 4*4*2 = 32
    int nbytes = N * sizeof(float);
    
    float *dx, *hx; 
    
    // 1. 分配内存 (注意：物理内存依然是线性的，只看总大小)
    cudaMalloc((void **)&dx, nbytes);
    hx = (float*) malloc(nbytes);
    
    // 初始化 host 数据
    printf("hx original (Linear View): \n");
    for (int i = 0; i < N; i++) {
        hx[i] = i;
    }

    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);

    // 2. 【重点】定义 3D 的 Grid 和 Block 维度
    // 假设我们想要 block 大小为 2x2x1
    dim3 blockSize(2, 2, 1); 
    
    // 计算 Grid 需要多少个 block 才能覆盖数据
    // (nx + blockSize.x - 1) / blockSize.x 这种写法是为了向上取整，防止除不尽
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, 
                  (ny + blockSize.y - 1) / blockSize.y, 
                  (nz + blockSize.z - 1) / blockSize.z);

    printf("Launch Config: Grid(%d,%d,%d), Block(%d,%d,%d)\n", 
            gridSize.x, gridSize.y, gridSize.z, 
            blockSize.x, blockSize.y, blockSize.z);

    // 3. 启动 Kernel
    sum_3d<<<gridSize, blockSize>>>(dx, nx, ny, nz);
    
    // 等待 GPU以此保证 printf 输出完整 (仅用于调试)
    cudaDeviceSynchronize();

    cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);
    
    printf("\nhx current (First 8 elements): \n");
    for (int i = 0; i < 8; i++) { // 只打印前8个看看效果
        printf("Index %d: %g\n", i, hx[i]);
    }

    cudaFree(dx);
    free(hx);
    return 0;
}