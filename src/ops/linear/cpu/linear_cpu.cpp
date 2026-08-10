#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t batch, size_t in_features, size_t out_features) {
    // Y = X @ W^T (+ b)
    // 例: X(2,4), W(3,4) → Y(2,3)
    //   Y[i,j] = sum_k X[i,k] * W[j,k]  （W 未转置存放，用 W[j,k] 当 W^T[k,j]）
    for (size_t i = 0; i < batch; i++) {
        for (size_t j = 0; j < out_features; j++) {
            float acc = 0.f;
            const T *x_row = in + i * in_features;
            const T *w_row = weight + j * in_features; // W 的第 j 行，长 in_features
            for (size_t k = 0; k < in_features; k++) {
                acc += llaisys::utils::cast<float>(x_row[k]) * llaisys::utils::cast<float>(w_row[k]);
            }
            if (bias != nullptr) {
                acc += llaisys::utils::cast<float>(bias[j]);
            }
            out[i * out_features + j] = llaisys::utils::cast<T>(acc);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch, size_t in_features, size_t out_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       bias ? reinterpret_cast<const float *>(bias) : nullptr,
                       batch, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
                       batch, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
                       batch, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
