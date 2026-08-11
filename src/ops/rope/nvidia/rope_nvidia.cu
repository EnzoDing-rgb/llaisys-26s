#include "rope_nvidia.cuh"

#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type,
          size_t seqlen, size_t nhead, size_t d, float theta) {
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops::nvidia
