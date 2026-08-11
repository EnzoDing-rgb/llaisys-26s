#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);

    // 例: index (50,), weight (512, 4096), out (50, 4096)
    ASSERT(index->ndim() == 1, "Embedding: index must be 1D.");
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2D.");
    ASSERT(out->ndim() == 2, "Embedding: out must be 2D.");
    ASSERT(index->shape()[0] == out->shape()[0], "Embedding: index length must equal out rows.");
    ASSERT(weight->shape()[1] == out->shape()[1], "Embedding: weight cols must equal out cols.");

    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be int64.");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");

    const size_t index_len = index->numel();
    const size_t embed_dim = weight->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index_len, embed_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index_len, embed_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index_len, embed_dim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
