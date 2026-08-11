#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // ---------- 形状约定 ----------
    // GQA 例: nhead=8, nkvhead=4 → 每 2 个 Q 共用一对 KV；out 仍是 8 路
    // cache 例: qlen=5, kvlen=11 → k/v 已含历史 6 + 当前 5
    // README「计算前连接 kvcache」: 拼接在调用方完成；这里 k/v 已是 [kvlen, ...]

    CHECK_SAME_DEVICE(attn_val, q, k, v);

    ASSERT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3 && attn_val->ndim() == 3,
           "SelfAttention: q/k/v/out must be 3D.");
    // q: [qlen,nhead,d], k: [kvlen,nkvhead,d], v: [kvlen,nkvhead,dv], out: [qlen,nhead,dv]
    ASSERT(q->shape()[0] == attn_val->shape()[0], "SelfAttention: qlen mismatch.");
    ASSERT(k->shape()[0] == v->shape()[0], "SelfAttention: kvlen mismatch between k and v.");
    ASSERT(q->shape()[0] <= k->shape()[0], "SelfAttention: qlen must be <= kvlen (cache+current).");
    ASSERT(q->shape()[1] == attn_val->shape()[1], "SelfAttention: nhead mismatch.");
    ASSERT(k->shape()[1] == v->shape()[1], "SelfAttention: nkvhead mismatch.");
    ASSERT(q->shape()[1] % k->shape()[1] == 0, "SelfAttention: nhead must be divisible by nkvhead.");
    ASSERT(q->shape()[2] == k->shape()[2], "SelfAttention: head dim d mismatch.");
    ASSERT(v->shape()[2] == attn_val->shape()[2], "SelfAttention: dv mismatch.");

    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: tensors must be contiguous.");

    const size_t qlen = q->shape()[0];
    const size_t kvlen = k->shape()[0];
    const size_t nhead = q->shape()[1];
    const size_t nkvhead = k->shape()[1];
    const size_t d = q->shape()[2];
    const size_t dv = v->shape()[2];

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(),
                                   qlen, kvlen, nhead, nkvhead, d, dv, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(),
                                   qlen, kvlen, nhead, nkvhead, d, dv, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(),
                                      qlen, kvlen, nhead, nkvhead, d, dv, scale);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
