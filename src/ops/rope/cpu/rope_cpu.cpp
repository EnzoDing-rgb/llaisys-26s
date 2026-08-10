#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seqlen, size_t nhead, size_t d,
           float theta) {
    // 例: shape (2,1,4) → seqlen=2, nhead=1, d=4, half=2
    // 每个 head 向量切成前后半:
    //   a = x[0 : half],  b = x[half : d]
    // 平面 j 旋转 (a[j], b[j])，角度 φ = p / theta^(2j/d)
    const size_t half = d / 2;
    const float d_f = static_cast<float>(d);

    for (size_t s = 0; s < seqlen; s++) {
        // 绝对位置，可能是 0,1 也可能是 512,513,...（KV cache 续写）
        const float p = static_cast<float>(pos_ids[s]);

        for (size_t h = 0; h < nhead; h++) {
            // 连续布局 [seqlen, nhead, d]：先 token，再 head，再 dim
            const T *x = in + (s * nhead + h) * d;
            T *y = out + (s * nhead + h) * d;

            for (size_t j = 0; j < half; j++) {
                // 例 d=4,θ=10000,p=2:
                //   j=0 → φ = 2 / 10000^0 = 2
                //   j=1 → φ = 2 / 10000^0.5 = 0.02
                const float freq = p / std::pow(theta, 2.f * static_cast<float>(j) / d_f);
                const float c = std::cos(freq);
                const float sphi = std::sin(freq);

                // 平面 j: 前半 a[j]=x[j] 配 后半 b[j]=x[half+j]
                const float a = llaisys::utils::cast<float>(x[j]);
                const float b = llaisys::utils::cast<float>(x[half + j]);

                // a' = a cosφ - b sinφ
                // b' = b cosφ + a sinφ
                y[j] = llaisys::utils::cast<T>(a * c - b * sphi);
                y[half + j] = llaisys::utils::cast<T>(b * c + a * sphi);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type,
          size_t seqlen, size_t nhead, size_t d, float theta) {
    const auto *pids = reinterpret_cast<const int64_t *>(pos_ids);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pids, seqlen,
                     nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(in), pids, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(in), pids, seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
