#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 例: in/out (2,1,4), pos_ids (2,) = [0,1], theta=10000
    // 含义: 对每个 token 的每个 head 向量做前后半 RoPE 旋转
    CHECK_SAME_DEVICE(out, in, pos_ids);

    // [seqlen, nhead, d]，连续；d 必须偶数才能前后对半
    ASSERT(in->ndim() == 3 && out->ndim() == 3, "RoPE: in/out must be 3D [seqlen,nhead,d].");
    ASSERT(in->shape() == out->shape(), "RoPE: in/out shape mismatch.");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D.");
    ASSERT(pos_ids->shape()[0] == in->shape()[0], "RoPE: pos_ids length must equal seqlen.");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64.");
    ASSERT(in->shape()[2] % 2 == 0, "RoPE: head dim d must be even.");

    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: tensors must be contiguous.");

    const size_t seqlen = in->shape()[0]; // 例 2 或 512
    const size_t nhead = in->shape()[1];  // 例 1 或 4
    const size_t d = in->shape()[2];      // 例 4 或 4096

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, d, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, d, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
