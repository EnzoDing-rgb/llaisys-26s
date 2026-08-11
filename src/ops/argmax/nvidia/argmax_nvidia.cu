#include "argmax_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

namespace llaisys::ops::nvidia {

// ---------------------------------------------------------------------------
// 设备端：把各 dtype 统一成 float 再比较（f16/bf16 直接比容易踩 NaN/精度坑）
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float argmax_as_float(T v);

template <>
__device__ __forceinline__ float argmax_as_float<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ float argmax_as_float<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float argmax_as_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// ---------------------------------------------------------------------------
// Pass 1：每个 block 在自己的「条带」里找 (max_value, max_index)
//
// 并行结构（例 numel=4096, threads=256, blocks=16）：
//   block 0 负责下标 0,16,32,... 再在 block 内归约
//   实际用的是「连续大段」划分：block b 负责 [b*chunk, (b+1)*chunk)
//   最后把每个 block 的答案写到 partial_val[b] / partial_idx[b]
//
// 平局规则与 CPU 相同：只有严格更大才更新 → 保留更小下标
// ---------------------------------------------------------------------------
template <typename T>
__global__ void argmax_partial_kernel(const T *vals, size_t numel, float *partial_val, size_t *partial_idx) {
    // 动态 shared memory：前半存 float 最优值，后半存对应下标
    extern __shared__ unsigned char smem[];
    float *s_val = reinterpret_cast<float *>(smem);
    size_t *s_idx = reinterpret_cast<size_t *>(s_val + blockDim.x);

    const size_t tid = threadIdx.x;
    const size_t bid = blockIdx.x;
    const size_t nthreads = blockDim.x;
    const size_t nblocks = gridDim.x;

    // 把 [0, numel) 切成 nblocks 段，本 block 负责一段（负载尽量均匀）
    const size_t chunk = (numel + nblocks - 1) / nblocks;
    const size_t begin = bid * chunk;
    const size_t end = min(begin + chunk, numel);

    float best = -FLT_MAX;
    size_t best_i = 0;

    // block 内再按 thread 跨步扫自己的段：thread 0 看 begin+0, begin+nthreads, ...
    for (size_t i = begin + tid; i < end; i += nthreads) {
        float v = argmax_as_float(vals[i]);
        if (v > best) {
            best = v;
            best_i = i;
        }
    }

    s_val[tid] = best;
    s_idx[tid] = best_i;
    __syncthreads();

    // 树形归约：256 → 128 → 64 → ... → 1，全程在 shared memory，避免反复打 global
    for (unsigned int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    // 每个 block 只写一个中间结果，供 Pass 2 再用
    if (tid == 0) {
        partial_val[bid] = s_val[0];
        partial_idx[bid] = s_idx[0];
    }
}

// ---------------------------------------------------------------------------
// Pass 2：把「每个 block 的局部最优」再归约成全局唯一答案，写回输出
// partial_len == Pass 1 的 gridDim.x
// ---------------------------------------------------------------------------
template <typename T>
__global__ void argmax_finalize_kernel(const T *vals, const float *partial_val,
                                       const size_t *partial_idx, size_t partial_len,
                                       int64_t *max_idx, T *max_val) {
    extern __shared__ unsigned char smem[];
    float *s_val = reinterpret_cast<float *>(smem);
    size_t *s_idx = reinterpret_cast<size_t *>(s_val + blockDim.x);

    const size_t tid = threadIdx.x;

    float best = -FLT_MAX;
    size_t best_i = 0;
    for (size_t i = tid; i < partial_len; i += blockDim.x) {
        float v = partial_val[i];
        if (v > best) {
            best = v;
            best_i = partial_idx[i];
        }
    }

    s_val[tid] = best;
    s_idx[tid] = best_i;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        // max_val 写原始 dtype 的元素；下标来自归约结果
        max_idx[0] = static_cast<int64_t>(s_idx[0]);
        max_val[0] = vals[s_idx[0]];
    }
}

// ---------------------------------------------------------------------------
// Host：选型 launch —— 多 block 吃满 SM，再用第二次 kernel 收拢
// ---------------------------------------------------------------------------
template <typename T>
void argmax_launch(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    const int threads = 256;
    // 每个 block 至少摊到一点活；上限 1024 避免 partial 过大
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 1024) {
        blocks = 1024;
    }

    const size_t shmem = threads * (sizeof(float) + sizeof(size_t));

    float *partial_val = nullptr;
    size_t *partial_idx = nullptr;
    ASSERT(cudaMalloc(&partial_val, blocks * sizeof(float)) == cudaSuccess,
           "argmax: cudaMalloc partial_val failed");
    ASSERT(cudaMalloc(&partial_idx, blocks * sizeof(size_t)) == cudaSuccess,
           "argmax: cudaMalloc partial_idx failed");

    argmax_partial_kernel<T><<<blocks, threads, shmem>>>(vals, numel, partial_val, partial_idx);
    argmax_finalize_kernel<T><<<1, threads, shmem>>>(vals, partial_val, partial_idx, static_cast<size_t>(blocks), max_idx, max_val);

    ASSERT(cudaDeviceSynchronize() == cudaSuccess, "argmax: cudaDeviceSynchronize failed");
    ASSERT(cudaFree(partial_val) == cudaSuccess, "argmax: cudaFree partial_val failed");
    ASSERT(cudaFree(partial_idx) == cudaSuccess, "argmax: cudaFree partial_idx failed");
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    auto *idx = reinterpret_cast<int64_t *>(max_idx);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_launch<float>(idx, reinterpret_cast<float *>(max_val),
                             reinterpret_cast<const float *>(vals), numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_launch<__half>(idx, reinterpret_cast<__half *>(max_val),
                              reinterpret_cast<const __half *>(vals), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_launch<__nv_bfloat16>(idx, reinterpret_cast<__nv_bfloat16 *>(max_val),
                                     reinterpret_cast<const __nv_bfloat16 *>(vals), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
