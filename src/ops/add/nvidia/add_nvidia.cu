#include "add_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// 一个 thread 处理一个下标 i；grid 盖住全部 numel 个元素
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    c[i] = a[i] + b[i];
}

// f16 / bf16：先升到 float 再加，再写回（和 CPU 侧 cast 策略一致，数值更稳）
__global__ void add_kernel_f16(__half *c, const __half *a, const __half *b, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float sum = __half2float(a[i]) + __half2float(b[i]);
    c[i] = __float2half(sum);
}

__global__ void add_kernel_bf16(__nv_bfloat16 *c, const __nv_bfloat16 *a, const __nv_bfloat16 *b, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    float sum = __bfloat162float(a[i]) + __bfloat162float(b[i]);
    c[i] = __float2bfloat16(sum);
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    // 256 threads/block；block 数 = 盖住 numel 所需的最少块数
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel);
        break;
    case LLAISYS_DTYPE_F16:
        // 内存布局与 llaisys::fp16_t（uint16）一致，按 CUDA __half 解释
        add_kernel_f16<<<blocks, threads>>>(
            reinterpret_cast<__half *>(c),
            reinterpret_cast<const __half *>(a),
            reinterpret_cast<const __half *>(b),
            numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel_bf16<<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(c),
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b),
            numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    // launch 异步；这里同步一次，保证返回前结果已写完（测例立刻对比）
    ASSERT(cudaDeviceSynchronize() == cudaSuccess, "add: cudaDeviceSynchronize failed");
}

} // namespace llaisys::ops::nvidia