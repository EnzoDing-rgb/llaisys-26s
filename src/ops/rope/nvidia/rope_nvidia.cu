#include "rope_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// =============================================================================
// 【算子 / 形状】
//   in, out : [seqlen, nhead, d]   d 必须为偶数；half = d/2
//   pos_ids : [seqlen]             每个 token 的绝对位置 p
//
//   每个 head 向量切成前后半 a=x[0:half], b=x[half:d]
//   对平面 j=0..half-1，用角度 φ = p / theta^(2j/d) 旋转一对 (a[j], b[j])：
//     a' = a cosφ - b sinφ
//     b' = b cosφ + a sinφ
//
// 【并行重点】
//   并行单元 = 一个旋转平面 (token s, head h, 平面 j)
//   不同 (s,h,j) 完全独立 → 一个 thread 负责一个（或多个跨步）平面即可
//   行间/平面间不需要归约，因此不用 SMEM
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
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                            size_t seqlen, size_t nhead, size_t d, float theta) {
    const size_t half = d / 2;
    // 总共有多少个独立旋转平面
    const size_t nplanes = seqlen * nhead * half;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const float d_f = static_cast<float>(d);

    // 【并行重点】把三维 (s,h,j) 摊成一维 plane_id，用 grid-stride 盖住
    for (size_t plane = blockIdx.x * blockDim.x + threadIdx.x; plane < nplanes; plane += stride) {
        // plane → (s, h, j)：先拆 j，再拆 h，最后 s（与连续布局对应）
        const size_t j = plane % half;
        const size_t tmp = plane / half;
        const size_t h = tmp % nhead;
        const size_t s = tmp / nhead;

        // 该 token 的绝对位置；KV cache 续写时可能是 512,513,... 而不是从 0 起
        const float p = static_cast<float>(pos_ids[s]);

        // φ = p / theta^(2j/d)
        const float freq = p / powf(theta, 2.f * static_cast<float>(j) / d_f);
        const float c = cosf(freq);
        const float sphi = sinf(freq);

        // 连续布局 [seqlen,nhead,d]：向量起点 = (s*nhead + h)*d
        const T *x = in + (s * nhead + h) * d;
        T *y = out + (s * nhead + h) * d;

        const float a = as_float(x[j]);
        const float b = as_float(x[half + j]);
        y[j] = from_float<T>(a * c - b * sphi);
        y[half + j] = from_float<T>(b * c + a * sphi);
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type,
          size_t seqlen, size_t nhead, size_t d, float theta) {
    const auto *pids = reinterpret_cast<const int64_t *>(pos_ids);
    const size_t nplanes = seqlen * nhead * (d / 2);

    const int threads = 256;
    int blocks = static_cast<int>((nplanes + threads - 1) / threads);
    if (blocks > 1024) {
        blocks = 1024;
    }
    if (blocks < 1) {
        blocks = 1;
    }

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pids,
            seqlen, nhead, d, theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(in), pids,
            seqlen, nhead, d, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<__nv_bfloat16><<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(in), pids,
            seqlen, nhead, d, theta);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    ASSERT(cudaDeviceSynchronize() == cudaSuccess, "rope: cudaDeviceSynchronize failed");
}

} // namespace llaisys::ops::nvidia
