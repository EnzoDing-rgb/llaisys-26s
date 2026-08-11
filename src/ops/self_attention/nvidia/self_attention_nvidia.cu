#include "self_attention_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace llaisys::ops::nvidia {

// =============================================================================
// 【算子 / 形状】
//   q   : [qlen,  nhead,   d ]
//   k   : [kvlen, nkvhead, d ]
//   v   : [kvlen, nkvhead, dv]
//   out : [qlen,  nhead,   dv]
//   cache_len = kvlen - qlen：k/v 前段是历史，后段对齐当前 query
//
// 【对每个 (i, q_head) 做什么】（与 CPU 同序）
//   1) 对每个 key 位置 j 算 score = scale * <q, k_j>；因果：j > i+cache_len → -inf
//   2) 对 scores 做稳定 softmax → weights
//   3) out = Σ_j weights[j] * v_j
//   GQA：kv_head = q_head / (nhead/nkvhead)
//
// 【并行重点】
//   外层：一个 CUDA block = 一个 (query token i, q_head) —— 这些对之间完全独立
//   块内：多 thread 分摊「各个 j 的点积 / softmax 归约 / 各个输出通道 c」
//   SMEM：只暂存本 block 的 scores（长度 kvlen）+ 归约便签（长度 blockDim）
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

// block 内对 red[0..blockDim) 做 max / sum 归约，结果在 red[0]
__device__ void block_reduce_max(float *red, unsigned int tid, unsigned int nthreads) {
    for (unsigned int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red[tid] = fmaxf(red[tid], red[tid + stride]);
        }
        __syncthreads();
    }
}

__device__ void block_reduce_sum(float *red, unsigned int tid, unsigned int nthreads) {
    for (unsigned int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void self_attention_kernel(T *out, const T *q, const T *k, const T *v,
                                      size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead,
                                      size_t d, size_t dv, float scale) {
    // 【并行重点】blockIdx.x 编码一对 (i, q_head)
    const size_t work = blockIdx.x;
    const size_t i = work / nhead;
    const size_t q_head = work % nhead;
    if (i >= qlen) {
        return;
    }

    const size_t q_heads_per_kv = nhead / nkvhead;
    const size_t kv_head = q_head / q_heads_per_kv;
    const size_t cache_len = kvlen - qlen;
    const size_t tid = threadIdx.x;
    const unsigned int nthreads = blockDim.x;

    // smem: [scores|weights 复用区 kvlen 个 float][归约缓冲 nthreads 个 float]
    extern __shared__ float smem[];
    float *scores = smem;
    float *red = smem + kvlen;

    const T *query_vec = q + (i * nhead + q_head) * d;

    // ---- 1) 各 thread 跨步负责若干 j，算注意力分数 ----
    for (size_t j = tid; j < kvlen; j += nthreads) {
        if (j > i + cache_len) {
            scores[j] = -INFINITY; // 因果遮罩：后面 softmax 权重为 0
            continue;
        }
        const T *key_vec = k + (j * nkvhead + kv_head) * d;
        float dot = 0.f;
        for (size_t c = 0; c < d; c++) {
            dot += as_float(query_vec[c]) * as_float(key_vec[c]);
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    // ---- 2a) 稳定 softmax：先求 max(scores)（跳过 -inf）----
    float local_max = -INFINITY;
    for (size_t j = tid; j < kvlen; j += nthreads) {
        local_max = fmaxf(local_max, scores[j]);
    }
    red[tid] = local_max;
    __syncthreads();
    block_reduce_max(red, tid, nthreads);
    const float max_score = red[0];
    __syncthreads();

    // ---- 2b) weights[j] = exp(score-max)，再归一化（原地写回 scores 当 weights）----
    float local_sum = 0.f;
    for (size_t j = tid; j < kvlen; j += nthreads) {
        float w = isinf(scores[j]) ? 0.f : expf(scores[j] - max_score);
        scores[j] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    __syncthreads();
    block_reduce_sum(red, tid, nthreads);
    const float sum_exp = red[0];
    __syncthreads();

    for (size_t j = tid; j < kvlen; j += nthreads) {
        scores[j] /= sum_exp; // 至此 scores[] 已是 softmax 权重
    }
    __syncthreads();

    // ---- 3) 各 thread 负责若干输出通道 c：加权求和 V ----
    T *out_vec = out + (i * nhead + q_head) * dv;
    for (size_t c = tid; c < dv; c += nthreads) {
        float acc = 0.f;
        for (size_t j = 0; j < kvlen; j++) {
            const T *value_vec = v + (j * nkvhead + kv_head) * dv;
            acc += scores[j] * as_float(value_vec[c]);
        }
        out_vec[c] = from_float<T>(acc);
    }
}

void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead,
                    size_t d, size_t dv, float scale) {
    // 【并行重点】grid = qlen * nhead 个 block，一对 (i,q_head) 一个 block
    const int threads = 256;
    const int blocks = static_cast<int>(qlen * nhead);
    const size_t shmem = kvlen * sizeof(float) + static_cast<size_t>(threads) * sizeof(float);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel<float><<<blocks, threads, shmem>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
            qlen, kvlen, nhead, nkvhead, d, dv, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel<__half><<<blocks, threads, shmem>>>(
            reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(q),
            reinterpret_cast<const __half *>(k), reinterpret_cast<const __half *>(v),
            qlen, kvlen, nhead, nkvhead, d, dv, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel<__nv_bfloat16><<<blocks, threads, shmem>>>(
            reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k), reinterpret_cast<const __nv_bfloat16 *>(v),
            qlen, kvlen, nhead, nkvhead, d, dv, scale);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    ASSERT(cudaDeviceSynchronize() == cudaSuccess, "self_attention: cudaDeviceSynchronize failed");
}

} // namespace llaisys::ops::nvidia
