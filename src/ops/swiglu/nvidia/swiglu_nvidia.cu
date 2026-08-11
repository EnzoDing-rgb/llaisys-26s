#include "swiglu_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// =============================================================================
// 【算子】out[i] = up[i] * SiLU(gate[i])，SiLU(g)=g/(1+exp(-g))
//
// 【并行重点】并行的是「元素下标 i」——每个 thread 独立算若干个 i 上的 SiLU*up。
//   - 没有归约、没有 block 内交换 → 不用 SMEM
//   - 和 add 同类（elementwise）；和 argmax（归约）/ embedding（按行 gather）都不同
// 【访存】每个 i：global 读 gate/up → 寄存器里用 float 算 → global 写 out
// 【dtype】as_float / from_float：只是「元素 ↔ float」转换，算仍在 float 里做
// =============================================================================

template <typename T>
__device__ __forceinline__ float as_float(T v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(v);
    } else {
        return __bfloat162float(v);
    }
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __float2half(v);
    } else {
        return __float2bfloat16(v);
    }
}

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    // 【并行重点】grid-stride：不同 thread 同时处理不同 i；numel 很大时再按 stride 多轮
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < numel; i += stride) {
        const float g = as_float(gate[i]);
        const float u = as_float(up[i]);
        const float silu = g / (1.f + expf(-g));
        out[i] = from_float<T>(u * silu);
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    const int threads = 256;
    // 【并行重点】blocks×threads = 同时开工的 thread 数上限；其余靠 stride 扫完
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (blocks > 1024) {
        blocks = 1024;
    }
    if (blocks < 1) {
        blocks = 1;
    }

    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up), numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(gate),
            reinterpret_cast<const __half *>(up), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<__nv_bfloat16><<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(gate),
            reinterpret_cast<const __nv_bfloat16 *>(up), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    ASSERT(cudaDeviceSynchronize() == cudaSuccess, "swiglu: cudaDeviceSynchronize failed");
}

} // namespace llaisys::ops::nvidia
