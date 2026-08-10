#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// 例: in/out 连续存成 [行0的 d 个数 | 行1的 d 个数 | ...]
// rows=512, d=4096, weight 长度 d；type 决定指针按 float/fp16/bf16 解释
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type,
              size_t rows, size_t d, float eps);
}
