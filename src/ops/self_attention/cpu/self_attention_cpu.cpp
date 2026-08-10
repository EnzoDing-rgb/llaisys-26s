#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *out, const T *q, const T *k, const T *v, size_t qlen, size_t kvlen,
                     size_t nhead, size_t nkvhead, size_t d, size_t dv, float scale) {
    // =====================================================================
    // 贯穿例子：nhead=8, nkvhead=4, d=dv=64, qlen=5, kvlen=11
    // （前 6 个 token 在 KV cache，当前段 5 个 query）
    //
    // ---------- q 的内存到底怎么排（最易懵的地方）----------
    // shape 约定是 [qlen, nhead, d] = [5, 8, 64]，行优先、连续存放。
    // 含义：先固定「第几个 query token」，再排完它的 8 个 head，每个 head 64 维。
    //
    // 内存从基址 q 起，一块一块往后摆（每块 64 个数）：
    //   块0:  token i=0, head0 的 64 维
    //   块1:  token i=0, head1 的 64 维
    //   ...
    //   块7:  token i=0, head7 的 64 维          ← 到这里，token0 用掉 8 块
    //   块8:  token i=1, head0 的 64 维
    //   块9:  token i=1, head1 的 64 维
    //   ...
    //
    // 所以：
    //   i      = 当前是第几个 query token（0..4）
    //   * 8    = 每个 token 占 8 个 head 块，跨到下一个 token 要跳 8 块
    //   + q_head = 再在本 token 内选第几个 head 块（0..7）
    //   * 64   = 每块有 d=64 个元素，把「块号」换成「元素偏移」
    //
    // 公式：偏移 = (i * nhead + q_head) * d = (i * 8 + q_head) * 64
    // 例：要取 q[token=1, head=3, :]
    //   先跨过 token0 的 8 块 → i*8 = 8
    //   再加本 token 的 head3 → 8+3 = 11
    //   再 *64 → 从基址起第 11*64 个元素开始，连续读 64 个数
    //
    // 一句话：一个 token 里并排 8 个 head；i 乘 8 是为了跳过前面那些 token 占掉的整段。
    // 存放顺序是「token → head → dim」：先写完 token0 的 head0..7，再写 token1 的 head0..7。
    //
    // k/v 同理，只是第二维是 nkvhead=4：
    //   偏移 = (j * 4 + kv_head) * 64
    // out 与 q 同布局：偏移 = (i * 8 + q_head) * 64
    //
    // GQA：q_head 0,1→kv0；2,3→kv1；4,5→kv2；6,7→kv3
    // =====================================================================

    // cache_len=6：k/v 下标 0..5 是历史，6..10 对齐当前 query 0..4
    const size_t cache_len = kvlen - qlen;
    // 每个 kv_head 服务多少个 q_head；此处 8/4=2
    const size_t q_heads_per_kv = nhead / nkvhead;

    // 长度 kvlen=11：对「当前某一个 (i, q_head)」暂存对各 key 的分数与权重
    std::vector<float> scores(kvlen);
    std::vector<float> weights(kvlen);

    // 外层：逐个 query head。先算完 head0 的所有 token，再 head1……
    for (size_t q_head = 0; q_head < nhead; q_head++) {
        // 整数除法做分组。例：q_head=0 → 0；q_head=1 → 0；q_head=2 → 1
        const size_t kv_head = q_head / q_heads_per_kv;

        // 内层：当前段里的每个 query 位置 i = 0..4
        for (size_t i = 0; i < qlen; i++) {
            // 取出 q[i, q_head, 0:d]。例 i=0,q_head=0 → q 开头连续 64 个数
            const T *query_vec = q + (i * nhead + q_head) * d;

            // ----------------------------------------------------------
            // 步骤 1：对每个 key 位置 j 算注意力分数，并做因果遮罩
            // 全局上，query i 对应位置 i+cache_len（例 i=0 → 全局 6）
            // 允许看到的 key：j = 0,1,...,i+cache_len（例 i=0 → j=0..6）
            // ----------------------------------------------------------
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < kvlen; j++) {
                // j 落在「当前 query 之后」的那段：分数记为 -inf，softmax 后权重为 0
                // 例 i=0：j=7,8,9,10 走这里
                if (j > i + cache_len) {
                    scores[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                // 取出 k[j, kv_head, :]。例 j=3, kv_head=0 → 第 3 个 token、第 0 对 K
                const T *key_vec = k + (j * nkvhead + kv_head) * d;

                // 标准点积 <query_vec, key_vec>，在 float 里累加（半精度也先升 float）
                float dot = 0.f;
                for (size_t c = 0; c < d; c++) {
                    dot += llaisys::utils::cast<float>(query_vec[c]) *
                           llaisys::utils::cast<float>(key_vec[c]);
                }

                // 乘 scale（通常 1/sqrt(d)），写入 scores[j]
                scores[j] = dot * scale;

                // 顺带跟踪最大值，供下一步稳定 softmax（减去 max 再 exp）
                if (scores[j] > max_score) {
                    max_score = scores[j];
                }
            }

            // ----------------------------------------------------------
            // 步骤 2：对 j=0..kvlen-1 做 softmax，得到 weights
            // weights[j] = exp(scores[j]-max) / sum_t exp(scores[t]-max)
            // -inf 的位置 exp 视为 0
            // ----------------------------------------------------------
            float sum_exp = 0.f;
            for (size_t j = 0; j < kvlen; j++) {
                if (scores[j] == -std::numeric_limits<float>::infinity()) {
                    weights[j] = 0.f;
                } else {
                    weights[j] = std::exp(scores[j] - max_score);
                    sum_exp += weights[j];
                }
            }
            // 归一化，使 sum_j weights[j] == 1（因果可见位置上）
            for (size_t j = 0; j < kvlen; j++) {
                weights[j] /= sum_exp;
            }

            // ----------------------------------------------------------
            // 步骤 3：用权重对 V 加权求和，写出 out[i, q_head, :]
            // 例 q_head=0 → 写 out 的第 0 路；q_head=1 → 写第 1 路
            // 两者可能共用 kv_head=0 的 V，但 weights 来自不同 Q，结果仍是两路
            // ----------------------------------------------------------
            T *out_vec = out + (i * nhead + q_head) * dv;
            for (size_t c = 0; c < dv; c++) {
                float acc = 0.f;
                for (size_t j = 0; j < kvlen; j++) {
                    // v[j, kv_head, c]；与上面 key 用同一个 kv_head
                    const T *value_vec = v + (j * nkvhead + kv_head) * dv;
                    acc += weights[j] * llaisys::utils::cast<float>(value_vec[c]);
                }
                out_vec[c] = llaisys::utils::cast<T>(acc);
            }
        } 
    }     
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nhead, size_t nkvhead,
                    size_t d, size_t dv, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                               qlen, kvlen, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(out),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v), qlen, kvlen, nhead,
                               nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(out),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v), qlen, kvlen, nhead,
                               nkvhead, d, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
