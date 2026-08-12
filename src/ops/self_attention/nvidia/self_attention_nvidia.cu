#include "self_attention_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace llaisys::ops::nvidia {

// =============================================================================
// 贯穿例子（和 CPU 版同一套数字，方便对照）：
//   nhead=8, nkvhead=4, d=dv=64, qlen=5, kvlen=11
//   → cache_len = 11-5 = 6（k/v 前 6 个是历史，后 5 个对齐当前 query）
//
// ---------- 1) 张量形状 ----------
//   q   : [qlen,  nhead,   d ] = [5, 8, 64]
//   k   : [kvlen, nkvhead, d ] = [11, 4, 64]
//   v   : [kvlen, nkvhead, dv] = [11, 4, 64]
//   out : [qlen,  nhead,   dv] = [5, 8, 64]
//
// ---------- 2) 内存怎么排（行优先，连续）----------
// q 存放顺序是「token → head → dim」：
//   块0: token0,head0 的 64 维
//   块1: token0,head1 的 64 维
//   ...
//   块7: token0,head7 的 64 维   ← token0 占满 8 块
//   块8: token1,head0 的 64 维
//   ...
// 取 q[q_pos, q_head, :] 的元素偏移：
//   (q_pos * nhead + q_head) * d = (q_pos * 8 + q_head) * 64
// 例：q_pos=1, q_head=3 → (1*8+3)*64 = 11*64
//
// k/v 同理，只是第二维是 nkvhead=4：
//   偏移 = (j * nkvhead + kv_head) * d = (j * 4 + kv_head) * 64
// out 与 q 同布局。
//
// ---------- 3) MHA / GQA / MQA（只影响「Q 用哪一路 KV」）----------
//   MHA：nkvhead == nhead，一路 Q 对应一路 KV（本例不是）
//   GQA：nhead > nkvhead > 1，若干路 Q 共用一路 KV（本例：8 Q / 4 KV）
//   MQA：nkvhead == 1，全部 Q 共用同一路 KV
// 映射（整数除法）：
//   q_heads_per_kv = nhead / nkvhead = 8/4 = 2
//   kv_head = q_head / 2
//   → q_head 0,1→kv0；2,3→kv1；4,5→kv2；6,7→kv3
// 注意：out 仍然是 nhead=8 路；共用 KV 只是读同一份 K/V，权重仍由各自 Q 算。
//
// ---------- 4) 算什么（对固定的一个 (q_pos, q_head)）----------
//   1) scores[j] = scale * <q, k_j>；若 j > q_pos+cache_len → -inf（因果）
//   2) softmax(scores) → 权重
//   3) out = Σ_j 权重[j] * v_j
//
// ---------- 5) GPU 怎么拆 ----------
//   外层：一个 CUDA block = 一个 (q_pos, q_head)，共 qlen*nhead 个 block
//   块内：多 thread 分摊 key 下标 j / softmax 归约 / 输出通道 c
//   SMEM：本 block 的 scores[kvlen] + reduce_scratch[nthreads]
// =============================================================================

template <typename T>
__device__ __forceinline__ float as_float(T v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(v);
    } else {
        return __bfloat162float(v);
    }
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __float2half(v);
    } else {
        return __float2bfloat16(v);
    }
}

// block 内：把每个 thread 交上来的局部值做树状 max/sum，最终答案在 buf[0]
__device__ void block_reduce_max(float *buf, unsigned int tid, unsigned int nthreads) {
    for (unsigned int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            buf[tid] = fmaxf(buf[tid], buf[tid + stride]);
        }
        __syncthreads();
    }
}

__device__ void block_reduce_sum(float *buf, unsigned int tid, unsigned int nthreads) {
    for (unsigned int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            buf[tid] += buf[tid + stride];
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void self_attention_kernel(T *out, const T *q, const T *k, const T *v,
                                      size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead,
                                      size_t d, size_t dv, float scale) {
    // -------------------------------------------------------------------------
    // 【装载】本 block 认领哪一个「独立问题」
    //
    // 一共 qlen*nhead 个问题，每个问题 = 「某个 query token 的某一路 Q head，
    // 对全部 key 做 attention」。用 blockIdx.x 一对一编号：
    //   work = blockIdx.x          ∈ [0, qlen*nhead)
    //   q_pos         = flat / nhead        ∈ [0, qlen)
    //   q_head          = flat % nhead        ∈ [0, nhead)
    //
    // 例 qlen=5, nhead=8 → 40 个 block：
    //   block 0  → (q_pos=0, q_head=0)
    //   block 1  → (q_pos=0, q_head=1)
    //   ...
    //   block 7  → (q_pos=0, q_head=7)
    //   block 8  → (q_pos=1, q_head=0)
    // -------------------------------------------------------------------------
    const size_t work = blockIdx.x;
    const size_t q_pos = work / nhead;
    const size_t q_head = work % nhead;
    if (q_pos >= qlen) {
        return;
    }

    // GQA/MQA：把 q_head 映射到要用的那一路 KV
    // 例 nhead=8,nkvhead=4 → 每 2 路 Q 共用 1 路 KV；q_head=5 → kv_head=2
    // 若 MQA（nkvhead=1）：所有 q_head 都得到 kv_head=0
    const size_t q_heads_per_kv = nhead / nkvhead;
    const size_t kv_head = q_head / q_heads_per_kv;

    // cache_len=6：k/v 下标 0..5 是历史；6..10 对齐当前 query 的 q_pos=0..4
    const size_t cache_len = kvlen - qlen;
    const size_t tid = threadIdx.x;           // 本 thread 在 block 内编号，0..255
    const unsigned int nthreads = blockDim.x; // 通常 256

    // SMEM 只服务「本 block 这一个 (q_pos,q_head)」：
    //   scores[0..kvlen)            —— 先放分数，后原地改成 softmax 权重
    //   reduce_scratch[0..nthreads) —— 块内求 max/sum 时每 thread 放局部结果
    extern __shared__ float smem[];
    float *scores = smem;
    float *reduce_scratch = smem + kvlen;

    // 取出 q[q_pos, q_head, 0:d]
    // 例 q_pos=0,q_head=0 → 从 q 开头连续读 d=64 个数
    const T *query_vec = q + (q_pos * nhead + q_head) * d;

    // -------------------------------------------------------------------------
    // 步骤 1：填 scores[j] = scale * <query_vec, k[j, kv_head, :]>
    //
    // 工作怎么分给 thread：
    //   for (j = tid; j < kvlen; j += nthreads)
    // 意思：thread tid 负责 key 位置 j = tid, tid+256, tid+512, ...
    //
    // 小例子 kvlen=11, nthreads=256：
    //   tid=0  → 只算 j=0（因为 0+256=256 >= 11，停）
    //   tid=1  → 只算 j=1
    //   ...
    //   tid=10 → 只算 j=10
    //   tid=11..255 → 循环一开始就不进，这段空转
    // 所以短 KV 时按 j 分活会浪费线程；长 KV（>256）时同一 thread 会多轮。
    // 固定 256 是实现简单；同一批 thread 后面还会在输出通道 c 上干活。
    //
    // 因果遮罩：query q_pos 在全局对应位置 q_pos+cache_len
    //   例 q_pos=0 → 全局位置 6，只允许看 j=0..6；j=7..10 记 -inf
    // -------------------------------------------------------------------------
    for (size_t j = tid; j < kvlen; j += nthreads) {
        if (j > q_pos + cache_len) {
            scores[j] = -INFINITY; // 后面 softmax 权重会变成 0
            continue;
        }
        // 取出 k[j, kv_head, :]；注意用的是映射后的 kv_head，不是 q_head
        // 例 j=3, kv_head=0 → 偏移 (3*4+0)*64
        const T *key_vec = k + (j * nkvhead + kv_head) * d;
        float dot = 0.f;
        // 单个 j 上的点积：本 thread 串行扫完 d 维（测例 d 很小）
        for (size_t c = 0; c < d; c++) {
            dot += as_float(query_vec[c]) * as_float(key_vec[c]);
        }
        scores[j] = dot * scale;
    }
    __syncthreads(); // 所有 thread 写完 scores 再往下做 softmax

    // -------------------------------------------------------------------------
    // 步骤 2a：稳定 softmax 先求 max(scores)
    // 每个 thread 先看自己分到的那些 j，得到 local_max，写入 reduce_scratch[tid]
    // 再树状归约，最终 max 在 reduce_scratch[0]
    // -------------------------------------------------------------------------
    float local_max = -INFINITY;
    for (size_t j = tid; j < kvlen; j += nthreads) {
        local_max = fmaxf(local_max, scores[j]);
    }
    reduce_scratch[tid] = local_max;
    __syncthreads();
    block_reduce_max(reduce_scratch, tid, nthreads);
    const float max_score = reduce_scratch[0];
    __syncthreads();

    // -------------------------------------------------------------------------
    // 步骤 2b：weights[j] = exp(score-max) / sum；原地写回 scores 当权重
    // -inf 位置视为权重 0
    // -------------------------------------------------------------------------
    float local_sum = 0.f;
    for (size_t j = tid; j < kvlen; j += nthreads) {
        float w = isinf(scores[j]) ? 0.f : expf(scores[j] - max_score);
        scores[j] = w;
        local_sum += w;
    }
    reduce_scratch[tid] = local_sum;
    __syncthreads();
    block_reduce_sum(reduce_scratch, tid, nthreads);
    const float sum_exp = reduce_scratch[0];
    __syncthreads();

    for (size_t j = tid; j < kvlen; j += nthreads) {
        scores[j] /= sum_exp; // 至此 scores[] 已是 softmax 权重，且可见位置和为 1
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // 步骤 3：out[q_pos, q_head, c] = Σ_j scores[j] * v[j, kv_head, c]
    //
    // 这里按输出通道 c 分活：thread tid 负责 c = tid, tid+256, ...
    // 例 dv=64：只有 tid=0..63 真正写输出；tid=64..255 空转
    // （和步骤 1 相反：短 kvlen 闲、短 dv 也闲——同一固定 256 的代价）
    //
    // GQA 提醒：q_head=0 和 q_head=1 都读同一份 V（kv_head=0），
    // 但各自有自己的 scores，所以 out 仍是两路不同结果。
    // -------------------------------------------------------------------------
    T *out_vec = out + (q_pos * nhead + q_head) * dv;
    for (size_t c = tid; c < dv; c += nthreads) {
        float acc = 0.f;
        for (size_t j = 0; j < kvlen; j++) {
            const T *value_vec = v + (j * nkvhead + kv_head) * dv;
            acc += scores[j] * as_float(value_vec[c]);
        }
        out_vec[c] = from_float<T>(acc);
    }
}

void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead,
                    size_t d, size_t dv, float scale) {
    // 【并行重点】grid = qlen * nhead 个 block，一对 (q_pos, q_head) 一个 block
    // 例 qlen=5,nhead=8 → 40 blocks；每 block 256 threads
    const int threads = 256;
    const int blocks = static_cast<int>(qlen * nhead);
    // 动态 SMEM = scores(kvlen) + reduce_scratch(threads)
    const size_t shmem = kvlen * sizeof(float) + static_cast<size_t>(threads) * sizeof(float);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel<float><<<blocks, threads, shmem>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
            qlen, kvlen, nhead, nkvhead, d, dv, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel<__half><<<blocks, threads, shmem>>>(
            reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(q),
            reinterpret_cast<const __half *>(k), reinterpret_cast<const __half *>(v),
            qlen, kvlen, nhead, nkvhead, d, dv, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel<__nv_bfloat16><<<blocks, threads, shmem>>>(
            reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k), reinterpret_cast<const __nv_bfloat16 *>(v),
            qlen, kvlen, nhead, nkvhead, d, dv, scale);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    ASSERT(cudaDeviceSynchronize() == cudaSuccess, "self_attention: cudaDeviceSynchronize failed");
}

} // namespace llaisys::ops::nvidia
