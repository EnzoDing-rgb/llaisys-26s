#include "rms_norm_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// =============================================================================
// 【算子 / 形状】
//   in, out : [rows, d]   例测例 (512, 4096)
//   weight  : [d]         所有行共用同一条「按列」scale
//   对每一行 i：
//     inv_rms = 1 / sqrt( mean(in[i,:]^2) + eps )
//     out[i,j] = in[i,j] * inv_rms * weight[j]
//
// 【并行重点】（和 swiglu「一 thread 一元素」不同！）
//   1) blockIdx.x = 行号 i     → 不同行并行
//   2) 行内：多 thread 一起扫这一行的 d 列
//        - 先各自累加自己的 sum(x^2) 片段
//        - 再用 SMEM 树状归约 → 得到这一行的 sum_sq → inv_rms
//        - 最后再跨步写 out[i,j]
//   为什么要归约？因为 inv_rms 依赖「整行」的均方，单个 thread 看不到整行就得协作。
//
// 【SMEM 在这里干什么】
//   只存「每个 thread 的局部 sum_sq」（长度=blockDim），供 block 内加总。
//   不是把整行 d=4096 的数据 cache 进 SMEM。
// =============================================================================

// 读写统一走 float 计算；用 if constexpr 收在一处，避免 swiglu 那种大段特化清单
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
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, size_t d, float eps) {
    // 【并行重点】一个 block = 一行
    const size_t row = blockIdx.x;
    const size_t tid = threadIdx.x;
    const T *x = in + row * d;
    T *y = out + row * d;

    // ---- Pass A：行内跨步，每人累加自己负责的那些 x^2 ----
    float partial = 0.f;
    for (size_t j = tid; j < d; j += blockDim.x) {
        float v = as_float(x[j]);
        partial += v * v;
    }

    // 【SMEM】便签本：s_sum[tid] = 该 thread 的 partial；稍后树状加总
    extern __shared__ float s_sum[];
    s_sum[tid] = partial;
    __syncthreads();

    // ---- Pass B：block 内归约 → s_sum[0] = 这一行的 sum_sq ----
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads(); // 下一层要读本层结果，必须对齐
    }

    // 所有 thread 都读同一个 inv_rms（s_sum[0] 在上一道 sync 后已稳定）
    // rsqrtf(z) ≈ 1/sqrt(z)，对应 CPU 的 1/sqrt(mean_sq+eps)
    const float inv_rms = rsqrtf(s_sum[0] / static_cast<float>(d) + eps);

    // ---- Pass C：再用同一套跨步，把归一化结果写回 out ----
    for (size_t j = tid; j < d; j += blockDim.x) {
        float v = as_float(x[j]);
        float w = as_float(weight[j]); // weight 按列 j，所有行共享
        y[j] = from_float<T>(v * inv_rms * w);
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type,
              size_t rows, size_t d, float eps) {
    // 【并行重点】gridDim = rows → 一行一个 block；256 thread 覆盖列维 d
    const int threads = 256;
    const int blocks = static_cast<int>(rows);
    const size_t shmem = threads * sizeof(float);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<float><<<blocks, threads, shmem>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight), d, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<__half><<<blocks, threads, shmem>>>(
            reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(in),
            reinterpret_cast<const __half *>(weight), d, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<__nv_bfloat16><<<blocks, threads, shmem>>>(
            reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const __nv_bfloat16 *>(weight), d, eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    ASSERT(cudaDeviceSynchronize() == cudaSuccess, "rms_norm: cudaDeviceSynchronize failed");
}

} // namespace llaisys::ops::nvidia
