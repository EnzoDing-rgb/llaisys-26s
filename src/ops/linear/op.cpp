#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 例: in(2,4), weight(3,4), out(2,3), bias(3,)  →  Y = X @ W^T + b
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }

    ASSERT(in->ndim() == 2 && weight->ndim() == 2 && out->ndim() == 2,
           "Linear: in/weight/out must be 2D.");
    // batch=2, in_features=4, out_features=3
    ASSERT(in->shape()[0] == out->shape()[0], "Linear: batch mismatch.");
    ASSERT(in->shape()[1] == weight->shape()[1], "Linear: in_features mismatch.");
    ASSERT(out->shape()[1] == weight->shape()[0], "Linear: out_features mismatch.");

    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: tensors must be contiguous.");

    if (bias) {
        // bias 长度 = out_features，例 (3,)
        ASSERT(bias->ndim() == 1 && bias->shape()[0] == out->shape()[1],
               "Linear: bias shape must be (out_features,).");
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    const size_t batch = in->shape()[0];
    const size_t in_features = in->shape()[1];
    const size_t out_features = weight->shape()[0];
    const std::byte *bias_ptr = bias ? bias->data() : nullptr;

    auto call_cpu = [&]() {
        cpu::linear(out->data(), in->data(), weight->data(), bias_ptr, out->dtype(),
                    batch, in_features, out_features);
    };

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return call_cpu();
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return call_cpu();
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
