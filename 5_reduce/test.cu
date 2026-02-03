#include<bits/stdc++.h>
#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"

const int N = 1<<6;
__global__ void reduce (float %*din, float *dout) {
    // 省略具体实现

    int stid= threadIdx.x + N * blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float shar_sum[N];

    shar_sum[tid] = din[stid];
    __syncthreads();

    for(int i = 1; i < blockDim.x; i = i<<1)
    {
        if((tid % (2*i) == 0))
            shar_sum[tid] += shar_sum[tid+i];
        __syncthreads();
    }

    if(tid == 0)
        dout[blockIdx.x] = shar_sum[0];
}


void softmax(float *input, float *output, int n) {
    // 省略具体实现
    int max_v = std::max(input,input+n);
    float sum = 0.0;
    for(int i = 0 ; i < n ; i++)
    {
        sum += std::exp(input[i] - max_v);
        output[i] = std::exp(input[i] - max_v)
    }

    for(int i = 0 ; i < n ; i++)
    {
        output[i] = output[i] / sum;
    }
}