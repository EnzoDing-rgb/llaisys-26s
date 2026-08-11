#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cstdint>

namespace llaisys::ops::nvidia {

// Embedding = 按行 gather：out[i, :] = weight[index[i], :]
// 没有算术，只是把 weight 里某一整行拷到 out 的第 i 行。
//
// 并行：一个 block 负责一行输出；block 内 threads 沿 embed_dim 跨步拷贝。
// 同一拍里 thread0..255 读/写的是行内连续地址 → 对 out/weight 行内访存友好。

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
