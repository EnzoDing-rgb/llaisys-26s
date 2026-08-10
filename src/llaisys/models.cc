#include "llaisys.h"
#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include "llaisys/ops.h"
#include <vector>
#include <cmath>
#include <cstring>

#include <cstddef>
#include <cstdlib>

__C {

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    llaisysDeviceType_t device;
    int device_id;

    llaisysTensor_t *k_cache;
    llaisysTensor_t *v_cache;
    size_t cache_len;
};

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {
    
    auto *model = new LlaisysQwen2Model();
    model->meta = *meta;
    model->device = device;
    model->device_id = device_ids[0];
    model->cache_len = 0;

    const auto dtype = meta->dtype;
    const int device_id = model->device_id;
    const size_t L = meta->nlayer;
    const size_t hs = meta->hs;
    const size_t di = meta->di;
    const size_t voc = meta->voc;
    const size_t kv_dim = meta->nkvh * meta->dh;

    size_t e2[2] = {voc, hs};
    model->weights.in_embed = tensorCreate(e2, 2, dtype, device, device_id);
    model->weights.out_embed = tensorCreate(e2, 2, dtype, device, device_id);

    size_t e1[1] = {hs};
    model->weights.out_norm_w = tensorCreate(e1, 1, dtype, device, device_id);

    model->weights.attn_norm_w = new llaisysTensor_t[L];
    model->weights.attn_q_w    = new llaisysTensor_t[L];
    model->weights.attn_q_b    = new llaisysTensor_t[L];
    model->weights.attn_k_w    = new llaisysTensor_t[L];
    model->weights.attn_k_b    = new llaisysTensor_t[L];
    model->weights.attn_v_w    = new llaisysTensor_t[L];
    model->weights.attn_v_b    = new llaisysTensor_t[L];
    model->weights.attn_o_w    = new llaisysTensor_t[L];
    model->weights.mlp_norm_w  = new llaisysTensor_t[L];
    model->weights.mlp_gate_w  = new llaisysTensor_t[L];
    model->weights.mlp_up_w    = new llaisysTensor_t[L];
    model->weights.mlp_down_w  = new llaisysTensor_t[L];
    model->k_cache = new llaisysTensor_t[L];
    model->v_cache = new llaisysTensor_t[L];
    
    size_t q_w[2] = {hs, hs};
    size_t q_b[1]  = {hs};
    size_t k_w[2]  = {kv_dim, hs};
    size_t k_b[1]  = {kv_dim};
    size_t o_w[2]  = {hs, hs};
    size_t gate[2] = {di, hs};
    size_t down[2] = {hs, di};
    size_t norm[1] = {hs};
    size_t kv_shape[3] = {meta->maxseq, meta->nkvh, meta->dh};

    for (size_t i = 0; i < L; ++i) {
        model->weights.attn_norm_w[i] =
            tensorCreate(norm, 1, dtype, device, device_id);
        model->weights.attn_q_w[i] =
            tensorCreate(q_w, 2, dtype, device, device_id);
        model->weights.attn_q_b[i] =
            tensorCreate(q_b, 1, dtype, device, device_id);
        model->weights.attn_k_w[i] =
            tensorCreate(k_w, 2, dtype, device, device_id);
        model->weights.attn_k_b[i] =
            tensorCreate(k_b, 1, dtype, device, device_id);
        model->weights.attn_v_w[i] =
            tensorCreate(k_w, 2, dtype, device, device_id);
        model->weights.attn_v_b[i] =
            tensorCreate(k_b, 1, dtype, device, device_id);
        model->weights.attn_o_w[i] =
            tensorCreate(o_w, 2, dtype, device, device_id);
        model->weights.mlp_norm_w[i] =
            tensorCreate(norm, 1, dtype, device, device_id);
        model->weights.mlp_gate_w[i] =
            tensorCreate(gate, 2, dtype, device, device_id);
        model->weights.mlp_up_w[i] =
            tensorCreate(gate, 2, dtype, device, device_id);
        model->weights.mlp_down_w[i] =
            tensorCreate(down, 2, dtype, device, device_id);
        model->k_cache[i] = 
            tensorCreate(kv_shape, 3, dtype, device, device_id);
        model->v_cache[i] = 
            tensorCreate(kv_shape, 3, dtype, device, device_id);
    }

    return model;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (!model) return;

    auto free_t = [](llaisysTensor_t t){
        if(t) tensorDestroy(t);
    };

    free_t(model->weights.in_embed);
    free_t(model->weights.out_embed);
    free_t(model->weights.out_norm_w);

    const size_t L = model->meta.nlayer;
    auto free_arr = [&](llaisysTensor_t *arr) {
        if (!arr) return;
        for (size_t i = 0; i < L; ++i) free_t(arr[i]);
        delete[] arr;
    };

    free_arr(model->weights.attn_norm_w);
    free_arr(model->weights.attn_q_w);
    free_arr(model->weights.attn_q_b);
    free_arr(model->weights.attn_k_w);
    free_arr(model->weights.attn_k_b);
    free_arr(model->weights.attn_v_w);
    free_arr(model->weights.attn_v_b);
    free_arr(model->weights.attn_o_w);
    free_arr(model->weights.mlp_norm_w);
    free_arr(model->weights.mlp_gate_w);
    free_arr(model->weights.mlp_up_w);
    free_arr(model->weights.mlp_down_w);
    free_arr(model->k_cache);
    free_arr(model->v_cache);

    delete model;
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    return &model->weights;
}

// 同形状连续 tensor 字节拷贝。
// 例: residual ← x，两边都是 [5, 1536] BF16，拷 5*1536*2 字节。
static void copy_tensor(llaisysTensor_t dst, llaisysTensor_t src) {
    size_t ndim = tensorGetNdim(src);
    std::vector<size_t> shape(ndim);
    tensorGetShape(src, shape.data());
    size_t numel = 1;
    for (size_t i = 0; i < ndim; ++i) numel *= shape[i];

    size_t esize = 2;
    switch (tensorGetDataType(src)) {
    case LLAISYS_DTYPE_F32: esize = 4; break;
    case LLAISYS_DTYPE_I64: esize = 8; break;
    default: esize = 2; break; // BF16 / F16
    }
    std::memcpy(tensorGetData(dst), tensorGetData(src), numel * esize);
}

int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken) {

    auto &meta = model->meta;
    auto &W = model->weights;
    const auto dtype = meta.dtype;
    const auto device = model->device;
    const int device_id = model->device_id;

    // 本模型典型值: hs=1536, nh=12, nkvh=2, dh=128, di=8960, L=28
    const size_t hs = meta.hs;
    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t di = meta.di;
    const size_t voc = meta.voc;
    const size_t L = meta.nlayer;
    const float eps = meta.epsilon;
    const float theta = meta.theta;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh)); // 1/sqrt(128)

    // 例: 上次 generate 留下 cache_len=200，这次新 prompt 只有 50 个 token
    //     → 50 <= 200，说明是新对话，cache 作废。
    if (ntoken <= model->cache_len) {
        model->cache_len = 0;
    }

    // 例 prefill: ntoken=50, cache_len=0 → past=0, n_new=50（整段 prompt 都要算）
    // 例 decode:  ntoken=51, cache_len=50 → past=50, n_new=1（只算刚生成的那 1 个）
    const size_t past = model->cache_len;
    const size_t n_new = ntoken - past;

    // ---- embedding：只嵌「新来的」token id ----
    // 例 n_new=5 → token_index 形状 (5,)
    // ================================================================
    // Embedding：把「新来的 token id」查成 hidden 向量。
    //
    // 数字例子（decode 一步）:
    //   past=50, n_new=1, hs=1536
    //   token_ids[50] = 91786
    //   → token_index 形状 (1,)，里面就一个数 91786
    //   → hidden 形状 (1, 1536)，一行 1536 维 BF16
    //   → Embedding 做的事: hidden[0] = in_embed[91786]
    //     （从词表矩阵第 91786 行拷一整行过来）
    //
    // Prefill 例子:
    //   past=0, n_new=50 → token_index 形状 (50,)，一次查 50 行
    //   hidden 形状 (50, 1536)，后面 28 层都在这块上算
    //
    // 为啥用 token_ids + past，而不是从头灌？
    //   前 past 个 token 的 K/V 已在 cache 里，它们的 embedding
    //   早就算过了；本轮只嵌「新尾巴」。
    // ================================================================
    size_t shape_token_index[1] = {n_new}; // 例 decode: {1}；prefill: {50}
    size_t shape_hidden[2] = {n_new, hs};  // 例 decode: {1,1536}

    // token_index: int64 一维，专门给 Embedding 当「行号」
    llaisysTensor_t token_index =
        tensorCreate(shape_token_index, 1, LLAISYS_DTYPE_I64, device, device_id);
    // hidden: 残差流起点；后面每层 Attention/MLP 都读写它
    llaisysTensor_t hidden =
        tensorCreate(shape_hidden, 2, dtype, device, device_id);

    // 例 past=50: 指针从 token_ids[50] 开始，只拷 n_new 个 id 进 token_index
    tensorLoad(token_index, token_ids + past);
    // 查表: hidden[t] = in_embed[ token_index[t] ]，in_embed 形状 (voc, hs)
    llaisysEmbedding(hidden, token_index, W.in_embed);
    tensorDestroy(token_index); // id 用完即可扔；hidden 要留给后面层


    // ================================================================
    // 给每个新 token 准备「绝对位置」，RoPE 要用。
    // 例 past=50, n_new=3 → rope_pos_host = {50,51,52}
    // 不能写成 {0,1,2}，否则角度和 cache 里旧的 K 对不齐。
    // ================================================================
    std::vector<int64_t> rope_pos_host(n_new);
    for (size_t i = 0; i < n_new; ++i)
        rope_pos_host[i] = static_cast<int64_t>(past + i);
    llaisysTensor_t rope_pos =
        tensorCreate(shape_token_index, 1, LLAISYS_DTYPE_I64, device, device_id);
    tensorLoad(rope_pos, rope_pos_host.data());

    // ---- 一层循环里反复用的临时缓冲（按本轮 n_new 开）----
    // 固定缩写:
    //   hs / nh / nkvh / dh / di / L / voc
    //   Q / K / V = query / key / value
    // 下面例子一律按: n_new=5, hs=1536, nh=12, nkvh=2, dh=128, di=8960

    // RMSNorm 后的 hidden，例 (5, hs)=(5,1536)
    llaisysTensor_t hidden_normed =
        tensorCreate(shape_hidden, 2, dtype, device, device_id);
    // 残差备份，做完 attn/mlp 再加回来
    llaisysTensor_t residual =
        tensorCreate(shape_hidden, 2, dtype, device, device_id);

    // Q/K/V 线性投影后的 2D（还没拆头）
    // q_flat: (5, hs)=(5,1536)；k_flat/v_flat: (5, nkvh*dh)=(5,256)
    llaisysTensor_t q_flat =
        tensorCreate(shape_hidden, 2, dtype, device, device_id);
    size_t shape_kv_flat[2] = {n_new, nkvh * dh};
    llaisysTensor_t k_flat =
        tensorCreate(shape_kv_flat, 2, dtype, device, device_id);
    llaisysTensor_t v_flat =
        tensorCreate(shape_kv_flat, 2, dtype, device, device_id);

    // 拆成多头 3D，给 RoPE / SelfAttention
    // q: (5, nh, dh)=(5,12,128)；k/v: (5, nkvh, dh)=(5,2,128)
    size_t shape_q[3] = {n_new, nh, dh};
    size_t shape_kv[3] = {n_new, nkvh, dh};
    llaisysTensor_t q = tensorCreate(shape_q, 3, dtype, device, device_id);
    llaisysTensor_t k = tensorCreate(shape_kv, 3, dtype, device, device_id);
    llaisysTensor_t v = tensorCreate(shape_kv, 3, dtype, device, device_id);

    // 注意力输出: attn (5,nh,dh) → attn_flat (5,hs) 再进 o_proj
    llaisysTensor_t attn =
        tensorCreate(shape_q, 3, dtype, device, device_id);
    llaisysTensor_t attn_flat =
        tensorCreate(shape_hidden, 2, dtype, device, device_id);

    // MLP: gate / up / swiglu_out 都是 (5, di)=(5,8960)
    size_t shape_mlp[2] = {n_new, di};
    llaisysTensor_t gate =
        tensorCreate(shape_mlp, 2, dtype, device, device_id);
    llaisysTensor_t up =
        tensorCreate(shape_mlp, 2, dtype, device, device_id);
    llaisysTensor_t swiglu_out =
        tensorCreate(shape_mlp, 2, dtype, device, device_id); // SiLU(gate)⊙up

    for (size_t i = 0; i < L; ++i) {
        // ========== 第 i 层 Attention ==========
        copy_tensor(residual, hidden);
        llaisysRmsNorm(hidden_normed, hidden, W.attn_norm_w[i], eps);

        // hidden_normed → q_flat / k_flat / v_flat
        llaisysLinear(q_flat, hidden_normed, W.attn_q_w[i], W.attn_q_b[i]);
        llaisysLinear(k_flat, hidden_normed, W.attn_k_w[i], W.attn_k_b[i]);
        llaisysLinear(v_flat, hidden_normed, W.attn_v_w[i], W.attn_v_b[i]);

        // 2D→3D：内存布局相同（hs=nh*dh），按字节灌入，等价 view
        // 例 q_flat (5,1536) → q (5,12,128)
        tensorLoad(q, tensorGetData(q_flat));
        tensorLoad(k, tensorGetData(k_flat));
        tensorLoad(v, tensorGetData(v_flat));

        // RoPE 只转 Q、K；V 不转。例 rope_pos={50,51,52}
        llaisysROPE(q, q, rope_pos, theta);
        llaisysROPE(k, k, rope_pos, theta);

        // 写入本层 KV cache 新槽。例 past=50,ntoken=53 → cache[50:53]
        llaisysTensor_t k_new = tensorSlice(model->k_cache[i], 0, past, ntoken);
        llaisysTensor_t v_new = tensorSlice(model->v_cache[i], 0, past, ntoken);
        copy_tensor(k_new, k);
        copy_tensor(v_new, v);
        tensorDestroy(k_new);
        tensorDestroy(v_new);

        // 注意力读全部 K/V。例 k_all=(53,nkvh,dh)，q 只有本轮 n_new 行
        llaisysTensor_t k_all = tensorSlice(model->k_cache[i], 0, 0, ntoken);
        llaisysTensor_t v_all = tensorSlice(model->v_cache[i], 0, 0, ntoken);
        llaisysSelfAttention(attn, q, k_all, v_all, scale);
        tensorDestroy(k_all);
        tensorDestroy(v_all);

        // attn (5,nh,dh) → attn_flat (5,hs) → o_proj → 加残差
        tensorLoad(attn_flat, tensorGetData(attn));
        llaisysLinear(hidden, attn_flat, W.attn_o_w[i], nullptr);
        llaisysAdd(hidden, residual, hidden);

        // ========== 第 i 层 MLP ==========
        copy_tensor(residual, hidden);
        llaisysRmsNorm(hidden_normed, hidden, W.mlp_norm_w[i], eps);
        llaisysLinear(gate, hidden_normed, W.mlp_gate_w[i], nullptr);
        llaisysLinear(up, hidden_normed, W.mlp_up_w[i], nullptr);
        llaisysSwiGLU(swiglu_out, gate, up);
        llaisysLinear(hidden, swiglu_out, W.mlp_down_w[i], nullptr);
        llaisysAdd(hidden, residual, hidden);
    }

    model->cache_len = ntoken; // 例算完 53 个，下次 past 从 53 起

    // ---- 只用最后一个新 token 的 hidden 预测下一个 id ----
    // 例 n_new=5 → 取 hidden 第 4 行，形状 (1,hs)
    llaisysTensor_t last_hidden = tensorSlice(hidden, 0, n_new - 1, n_new);
    size_t shape_one_row[2] = {1, hs};
    llaisysTensor_t last_normed =
        tensorCreate(shape_one_row, 2, dtype, device, device_id);
    llaisysRmsNorm(last_normed, last_hidden, W.out_norm_w, eps);

    // lm_head: (1,hs) → (1,voc)
    size_t shape_logits_2d[2] = {1, voc};
    llaisysTensor_t logits_2d =
        tensorCreate(shape_logits_2d, 2, dtype, device, device_id);
    llaisysLinear(logits_2d, last_normed, W.out_embed, nullptr);

    // Argmax 要 1D：(1,voc) 字节拷成 (voc,)
    size_t shape_vocab[1] = {voc};
    llaisysTensor_t logits_1d =
        tensorCreate(shape_vocab, 1, dtype, device, device_id);
    tensorLoad(logits_1d, tensorGetData(logits_2d));

    size_t shape_scalar[1] = {1};
    llaisysTensor_t max_idx =
        tensorCreate(shape_scalar, 1, LLAISYS_DTYPE_I64, device, device_id);
    llaisysTensor_t max_val =
        tensorCreate(shape_scalar, 1, dtype, device, device_id);
    llaisysArgmax(max_idx, max_val, logits_1d);
    int64_t next_id = *reinterpret_cast<int64_t *>(tensorGetData(max_idx));

    tensorDestroy(last_hidden);
    tensorDestroy(last_normed);
    tensorDestroy(logits_2d);
    tensorDestroy(logits_1d);
    tensorDestroy(max_idx);
    tensorDestroy(max_val);
    tensorDestroy(hidden);
    tensorDestroy(rope_pos);
    tensorDestroy(hidden_normed);
    tensorDestroy(residual);
    tensorDestroy(q_flat);
    tensorDestroy(k_flat);
    tensorDestroy(v_flat);
    tensorDestroy(q);
    tensorDestroy(k);
    tensorDestroy(v);
    tensorDestroy(attn);
    tensorDestroy(attn_flat);
    tensorDestroy(gate);
    tensorDestroy(up);
    tensorDestroy(swiglu_out);

    return next_id;
}

} // extern "C"
