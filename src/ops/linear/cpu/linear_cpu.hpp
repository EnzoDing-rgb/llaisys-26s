#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// bias == nullptr 表示无偏置
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch, size_t in_features, size_t out_features);
}
