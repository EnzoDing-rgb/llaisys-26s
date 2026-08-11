#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cstdint>

namespace llaisys::ops::nvidia {

// =============================================================================
// 【算子】out[row, :] = weight[index[row], :]  （按行 gather，只拷贝、无算术）
//
// 【并行重点】两层：
//   1) blockIdx.x = 输出行号 row  → 不同行之间并行
//   2) threadIdx.x 沿 embed_dim 跨步 → 同一行内多 thread 并行搬列
// 例 embed_dim=4096、threads=256 → 每 thread 大约搬 16 个元素（步长 blockDim）
// 【访存】dst[col]=src[col]：global→寄存器→global；行内连续下标利于合并
// =============================================================================

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                 size_t index_len, size_t embed_dim) {
    // 每个 block 对应输出的一行 i
    const size_t row = blockIdx.x;
    if (row >= index_len) {
        return;
    }

    // index[row] = 词表行号，例 3 → 去 weight 第 3 行
    const int64_t src_row = index[row];
    const T *src = weight + static_cast<size_t>(src_row) * embed_dim;
    T *dst = out + row * embed_dim;

    // 一行很长（测例里 embed_dim=4096）：多 thread 把这一行打满
    for (size_t col = threadIdx.x; col < embed_dim; col += blockDim.x) {
        dst[col] = src[col];
    }
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t index_len, size_t embed_dim) {
    const auto *idx = reinterpret_cast<const int64_t *>(index);
    // 一行一个 block；256 thread 扫列（4096/256=16 步即可扫完一行）
    const int threads = 256;
    const int blocks = static_cast<int>(index_len);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<float *>(out), idx, reinterpret_cast<const float *>(weight),
            index_len, embed_dim);
        break;
    case LLAISYS_DTYPE_F16:
        // fp16/bf16 都是 2 字节元素；用 uint16_t 做按元素拷贝即可（只搬比特）
        embedding_kernel<uint16_t><<<blocks, threads>>>(
            reinterpret_cast<uint16_t *>(out), idx, reinterpret_cast<const uint16_t *>(weight),
            index_len, embed_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<uint16_t><<<blocks, threads>>>(
            reinterpret_cast<uint16_t *>(out), idx, reinterpret_cast<const uint16_t *>(weight),
            index_len, embed_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    ASSERT(cudaDeviceSynchronize() == cudaSuccess, "embedding: cudaDeviceSynchronize failed");
}

} // namespace llaisys::ops::nvidia
