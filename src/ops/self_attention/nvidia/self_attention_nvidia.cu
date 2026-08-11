#include "self_attention_nvidia.cuh"

#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead,
                    size_t d, size_t dv, float scale) {
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops::nvidia
