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
//   并行的「一件活」= 一个旋转平面 (s, h, j)，同时动两个元素 x[j] 与 x[half+j]
//   因此把活摊成一维时，逻辑形状是 [seqlen, nhead, half]，最内维是 half 不是 d
//   （若按元素 [seqlen,nhead,d] 编号，会和「一对一算」错位）
//   不同 (s,h,j) 独立 → 无需归约 / 无需 SMEM
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
    // ---------------------------------------------------------------------
    // plane 编址的逻辑形状是 [seqlen, nhead, half]，不是 [seqlen, nhead, d]。
    //
    // 原因：并行的「一件活」= 旋转一对 (x[j], x[half+j])。
    //   一对已经覆盖头向量里的 2 个元素，所以每个 head 只有 half 件活，不是 d 件。
    //   若把活按元素数 d 来编号，会和「一对一算」对不上。
    //
    // 因此拆坐标时，最内维长度用 half：
    //   planes_per_token = nhead * half   （≈ 你直觉里的 nhead*(d/2)，不是 nhead*d）
    //   s = plane / planes_per_token
    //   余下再拆 h、j
    // ---------------------------------------------------------------------
    const size_t planes_per_head = half;
    const size_t planes_per_token = nhead * planes_per_head;
    const size_t nplanes = seqlen * planes_per_token;

    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const float d_f = static_cast<float>(d);

    // 【并行重点】plane 遍历所有可并行的旋转对；grid_stride 让 thread 多轮接下一批
    for (size_t plane = blockIdx.x * blockDim.x + threadIdx.x; plane < nplanes; plane += grid_stride) {
        // 按 [seqlen, nhead, half] 行主序从 plane 还原 (s, h, j)
        const size_t s = plane / planes_per_token;   // 落在第几个token
        const size_t rem = plane % planes_per_token; // 落在该 token 内的平面编号
        const size_t h = rem / planes_per_head;      // 第几个 head
        const size_t j = rem % planes_per_head;      // 该 head 内第几个旋转平面

        // 该 token 的绝对位置；KV cache 续写时可能是 512,513,... 
        const float p = static_cast<float>(pos_ids[s]);

        // φ = p / theta^(2j/d)
        const float freq = p / powf(theta, 2.f * static_cast<float>(j) / d_f);
        const float c = cosf(freq);
        const float sphi = sinf(freq);

        // 连续布局 [seqlen,nhead,d]：向量起点 = (s*nhead + h)*d
        const T *x = in + (s * nhead + h) * d;
        T *y = out + (s * nhead + h) * d;

        // 平面 j：同时读写前后半各一个元素
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
