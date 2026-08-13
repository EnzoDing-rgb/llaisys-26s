#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 测试里两组形状:
    //   (1, 4)     → rows=1,   d=4
    //   (512, 4096)→ rows=512, d=4096 ，weight 永远是 (d,)，例 (4096,)
    // 语义: 对每一行做 RMS 归一化，再逐元素 × weight（scale，不是矩阵乘）

    CHECK_SAME_DEVICE(out, in, weight);

    // in/out 必须是二维表: [行][特征]
    ASSERT(in->ndim() == 2 && out->ndim() == 2, "RMSNorm: in/out must be 2D.");
    // weight 是一条长度为 d 的向量，不是 (rows,d) 矩阵
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1D.");
    // 例: in 是 (512,4096)，out 也必须是 (512,4096)
    ASSERT(in->shape() == out->shape(), "RMSNorm: in/out shape mismatch.");
    // 例: weight.shape[0]==4096 必须等于 in 的列数 d
    ASSERT(weight->shape()[0] == in->shape()[1], "RMSNorm: weight length must equal d.");

    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    // 连续才能用「基址 + i*d + j」这种线性下标；有洞的话内核会读错位置
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm: tensors must be contiguous.");

    // 从 shape 拆出内核需要的两个整数（和 embedding 传 N/D 同一套路）
    const size_t rows = in->shape()[0]; // 例 512
    const size_t d = in->shape()[1];    // 例 4096；一行多长、W 多长

    // 把 Tensor 剥成: 数据指针 + dtype + rows + d + eps，交给 cpu 内核
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), rows, d, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), rows, d, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR: // 天数复用同一 CUDA kernel
#endif
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), rows, d, eps);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
