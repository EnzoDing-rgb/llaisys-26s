#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t d, float eps) {
    // 跑通测试里的小例子: shape (1,4)，一行四个数
    //   in  = [2,  2, -2,  2]
    //   weight 假设 = [1, 1, 1, 1]
    //   eps = 1e-5
    //
    // 公式 (对这一行):
    //   mean_sq = (2²+2²+(-2)²+2²)/4 = 4
    //   inv_rms = 1 / sqrt(mean_sq + eps) ≈ 1/2 = 0.5
    //   out[j]  = in[j] * inv_rms * weight[j]
    //          → [1, 1, -1, 1]

    for (size_t i = 0; i < rows; i++) {
        // 连续布局: 第 i 行起点 = 基址 + i*d
        // 例 i=0,d=4 → x 指向 in[0..3]；若还有第 2 行，i=1 → in[4..7]
        const T *x = in + i * d;
        T *y = out + i * d;

        // ---- 第一趟: 只算这一行的 sum(x^2)，得到 inv_rms ----
        // 用 float 累加，避免 F16/BF16 直接平方精度太差
        float sum_sq = 0.f;
        for (size_t j = 0; j < d; j++) {
            // cast: 若 T=fp16，先变成 float 再算；T=float 则几乎原样
            float v = llaisys::utils::cast<float>(x[j]);
            sum_sq += v * v; // 例: 依次累加 4,4,4,4 → sum_sq=16
        }
        // sum_sq/d = 均方；再 +eps 防止全 0 时除零
        // 例: 16/4 + 1e-5 = 4.00001 → sqrt ≈ 2 → inv_rms ≈ 0.5
        // 存「倒数」是为了第二趟做乘法而不是除法
        const float inv_rms = 1.f / std::sqrt(sum_sq / static_cast<float>(d) + eps);

        // ---- 第二趟: 写 out；weight 是按「列/特征维」共享的，所有行共用同一条 W ----
        // 例: y[0]=2*0.5*1=1, y[1]=2*0.5*1=1, y[2]=(-2)*0.5*1=-1, y[3]=2*0.5*1=1
        for (size_t j = 0; j < d; j++) {
            float v = llaisys::utils::cast<float>(x[j]);
            float w = llaisys::utils::cast<float>(weight[j]); // W 长度=d，下标也是 j
            // 最后 cast<T>: float 结果写回 F16/BF16/F32
            y[j] = llaisys::utils::cast<T>(v * inv_rms * w);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type,
              size_t rows, size_t d, float eps) {
    // 入口是 std::byte*（不懂类型的裸内存）；按 dtype 转成真正的元素指针再进模板
    // 三路都调同一个 rms_norm_，只是 T 不同
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), rows, d, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), rows, d, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), rows, d, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
