#include "rms_norm_nvidia.cuh"

#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type,
              size_t rows, size_t d, float eps) {
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops::nvidia
