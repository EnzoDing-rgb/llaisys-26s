#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t index_len, size_t embed_dim) {
    // 只搬比特，不算术；F32/F16/BF16 都走同一套按行 memcpy
    switch (type) {
    case LLAISYS_DTYPE_F32:
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    const auto *idx = reinterpret_cast<const int64_t *>(index);
    const size_t row_bytes = embed_dim * llaisys::utils::dsize(type);

    for (size_t i = 0; i < index_len; i++) {
        // 例: idx[i]=3, embed_dim=4096 → 拷 weight 第 3 整行到 out 第 i 行
        const std::byte *src = weight + static_cast<size_t>(idx[i]) * row_bytes;
        std::byte *dst = out + i * row_bytes;
        std::memcpy(dst, src, row_bytes);
    }
}
} // namespace llaisys::ops::cpu
