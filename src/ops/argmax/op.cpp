#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.cuh"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 三个张量必须在同一设备，例: 都在 CPU，不能 vals 在 CPU、max_val 在 GPU
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    // 作业约定: vals 是一维，例 shape (4096,)
    ASSERT(vals->ndim() == 1, "Argmax: vals must be 1D.");

    // 输出各只有一个元素，用来装「最大值的下标」和「最大值本身」，例 shape (1,)
    ASSERT(max_idx->ndim() == 1 && max_idx->numel() == 1, "Argmax: max_idx must be length-1.");
    ASSERT(max_val->ndim() == 1 && max_val->numel() == 1, "Argmax: max_val must be length-1.");
    
    // 最大值类型跟输入一致，例 vals 是 f16 → max_val 也是 f16
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    // 下标用 int64 存，例 max_idx[0] = 42
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be int64.");
    // 先只支持连续内存，方便内核按 i=0..numel-1 线性扫
    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(),
           "Argmax: all tensors must be contiguous.");

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR: // 天数复用同一 CUDA kernel
#endif
        return nvidia::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops

