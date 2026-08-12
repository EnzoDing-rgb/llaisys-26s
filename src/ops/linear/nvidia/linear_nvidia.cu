#include "linear_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// =============================================================================
// 【目标】做一次线性层：Y = X @ W^T + b（bias 可空）
//
// 贯穿小例子（与 CPU / 测例一致）：
//   in     shape = [batch, in_features]      = [2, 4]
//   weight shape = [out_features, in_features] = [3, 4]
//   bias   shape = [out_features]            = [3]   （也可以是 nullptr）
//   out    shape = [batch, out_features]     = [2, 3]
//
// 含义：
//   X 有 2 行样本，每行 4 维特征
//   W 有 3 个输出通道，每个通道对应一条长 4 的权重（W 按「未转置」存放）
//   结果 Y 是 2 行、每行 3 维
//
// 公式（对每个输出格子）：
//   Y[i, j] = Σ_{k=0..3} X[i, k] * W[j, k]  (+ bias[j])
// 例：Y[1, 2] = X[1,0]*W[2,0] + X[1,1]*W[2,1] + X[1,2]*W[2,2] + X[1,3]*W[2,3] + b[2]
//
// ---------- 内存怎么排（行优先）----------
// in / X：按行排，一行 in_features=4 个数
//   [ X行0的4个数 | X行1的4个数 ]
//   取 X 第 i 行起点：in + i * 4
//
// weight / W：也按行排，一行 in_features=4 个数；一共 out_features=3 行
//   [ W行0的4个数 | W行1的4个数 | W行2的4个数 ]
//   取 W 第 j 行起点：weight + j * 4
//   （PyTorch linear 要的是 X @ W^T，这里不显式转置，直接用「W 的第 j 行」去点乘）
//
// out / Y：按行排，一行 out_features=3 个数
//   [ Y行0的3个数 | Y行1的3个数 ]
//   格子 (i,j) 的扁平下标：idx = i * 3 + j
//
// ---------- 【并行重点】----------
//   独立输出一共 batch*out_features = 2*3 = 6 个格子 → 一 thread 负责一个 (i,j)
//   每个 thread 内部串行扫 k（in_features）：点积必须累加，并行 k 还得再 reduce
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
__global__ void linear_kernel(T *out, const T *in, const T *weight, const T *bias,
                              size_t batch, size_t in_features, size_t out_features) {
    // -------------------------------------------------------------------------
    // 【装载】本 thread 认领哪一个输出格子 Y[i, j]
    //
    // 例 batch=2, out_features=3 → total=6，idx 与 (i,j) 对应：
    //   idx 0 → (i=0, j=0)
    //   idx 1 → (i=0, j=1)
    //   idx 2 → (i=0, j=2)
    //   idx 3 → (i=1, j=0)
    //   idx 4 → (i=1, j=1)
    //   idx 5 → (i=1, j=2)
    // -------------------------------------------------------------------------
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = batch * out_features;
    if (idx >= total) {
        return;
    }

    const size_t i = idx / out_features; // 第几行样本；例 idx=4, out_features=3 → i=1
    const size_t j = idx % out_features; // 第几个输出通道；例 idx=4 → j=1

    // 例 i=1, in_features=4 → x_row 指向 X 第 1 行开头（跳过前 4 个数）
    const T *x_row = in + i * in_features;
    // 例 j=1, in_features=4 → w_row 指向 W 第 1 行开头（跳过前 4 个数）
    // 这一行就是「输出通道 j」的全部权重，长度 = in_features
    const T *w_row = weight + j * in_features;

    // -------------------------------------------------------------------------
    // 【计算】Y[i,j] = <X[i,:], W[j,:]> (+ bias[j])
    // 串行扫 k=0..in_features-1；例 in_features=4 就加 4 次
    // -------------------------------------------------------------------------
    float acc = 0.f;
    for (size_t k = 0; k < in_features; k++) {
        acc += as_float(x_row[k]) * as_float(w_row[k]);
    }
    // 例 j=1 → 加上 bias 的第 1 个元素；无 bias 时指针为 nullptr，跳过
    if (bias != nullptr) {
        acc += as_float(bias[j]);
    }
    // 例 idx=4 → 写入 out 里第 4 个格子，即 Y[1,1]
    out[idx] = from_float<T>(acc);
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch, size_t in_features, size_t out_features) {
    // 【并行重点】grid 盖住全部输出格子
    // 例 batch=2, out_features=3 → total=6 → 1 个 block（256 threads）就够，多出来的 thread 早退
    // 大测例 batch=512, out_features=4096 → total=2,097,152 → 需要很多 block
    const int threads = 256;
    const size_t total = batch * out_features;
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        linear_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            bias ? reinterpret_cast<const float *>(bias) : nullptr, batch, in_features,
            out_features);
        break;
    case LLAISYS_DTYPE_F16:
        linear_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(in),
            reinterpret_cast<const __half *>(weight),
            bias ? reinterpret_cast<const __half *>(bias) : nullptr, batch, in_features,
            out_features);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_kernel<__nv_bfloat16><<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            bias ? reinterpret_cast<const __nv_bfloat16 *>(bias) : nullptr, batch, in_features,
            out_features);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    ASSERT(cudaDeviceSynchronize() == cudaSuccess, "linear: cudaDeviceSynchronize failed");
}

} // namespace llaisys::ops::nvidia
