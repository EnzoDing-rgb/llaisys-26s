"""Qwen2 的 Python 外壳。

整条链路（从磁盘到能推理）长这样：

  磁盘上的模型目录
    ├── config.json          ← 超参数（层数、头数、词表大小……）
    └── *.safetensors        ← 真正的权重数值（bf16 字节）
           │
           ▼
  本文件 Qwen2.__init__
    1. 把 config 填进 LlaisysQwen2Meta（一张「尺寸说明书」）
    2. 调 C API Create：C++ 按说明书 malloc 出空 tensor 壳
    3. 调 C API Weights：拿到「壳」的指针表
    4. 按 safetensors 的名字，把权重字节 memcpy 进对应壳
           │
           ▼
  libllaisys.so 里的 C++ 模型对象（以后 Infer / generate 用它）

Python 这边故意不自己算 matmul / attention：
权重进 C++ 之后，前向完全走我们已经写好的 ops（embedding / linear /
rms_norm / rope / self_attention / swiglu / argmax）。
"""

from typing import Sequence
from pathlib import Path
import json
import re
from ctypes import c_int, c_int64, c_void_p, byref

import safetensors

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    LlaisysQwen2Meta,
)


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # model_path 例:
        #   /home/.../models/DeepSeek-R1-Distill-Qwen-1.5B
        # 里面至少要有 config.json 和若干 .safetensors。
        model_path = Path(model_path)

        # ================================================================
        # 步骤 1：读 config.json → 填 LlaisysQwen2Meta
        #
        # 为什么要这一步？
        #   C++ Create 时还看不到任何权重文件。它只知道「要造一个多大的
        #   模型」：多少层、hidden 多宽、词表多大……这些数字全在 config 里。
        #   Meta 就是把这些数字打包成 C 结构体，一次性交给 Create。
        #
        # 对本仓库那份 1.5B 模型，典型值是：
        #   hidden_size=1536, num_attention_heads=12, num_key_value_heads=2,
        #   num_hidden_layers=28, intermediate_size=8960, vocab_size=151936
        # ================================================================
        with open(model_path / "config.json") as f:
            cfg = json.load(f)

        hs = cfg["hidden_size"]              # 隐藏维，每个 token 的向量长度
        nh = cfg["num_attention_heads"]      # 注意力头数（Q 的头数）

        meta = LlaisysQwen2Meta(
            # 权重用 bf16 存；和 safetensors 里的 dtype 一致。
            dtype=DataType.BF16,

            # Transformer block 堆叠层数。每层一套 attn + mlp 权重。
            # 例: 28 → attn_q_w 是长度为 28 的指针数组。
            nlayer=cfg["num_hidden_layers"],

            # 残差流宽度。embedding 输出、每层输入输出都是 [seq, hs]。
            hs=hs,

            # Q 头数。self_attention 里 q 的形状是 [qlen, nh, dh]。
            nh=nh,

            # KV 头数。Qwen2 用 GQA：nkvh < nh，若干 Q 头共用一组 K/V。
            # 例: nh=12, nkvh=2 → 每 6 个 Q 头共享 1 组 KV。
            nkvh=cfg["num_key_value_heads"],

            # 每个头的维度。必须整除：hs = nh * dh。
            # 例: 1536 / 12 = 128。RoPE 也在这个 dh 维上转。
            dh=hs // nh,

            # MLP 中间维（gate/up 投影的输出宽）。
            # SwiGLU: gate, up : [hs] → [di]；down : [di] → [hs]。
            # 例: di=8960。
            di=cfg["intermediate_size"],

            # 位置编码 / KV cache 能支撑的最大序列长。
            # Create 时按这个预分配 K/V cache 槽位上限。
            maxseq=cfg["max_position_embeddings"],

            # 词表大小。in_embed / out_embed 的行数都是 voc。
            voc=cfg["vocab_size"],

            # RMSNorm 的 eps，防除零。公式: x * rsqrt(mean(x^2) + eps) * w
            epsilon=cfg["rms_norm_eps"],

            # RoPE 的基频 θ。角度 φ = pos / θ^(2j/dh)。
            theta=float(cfg["rope_theta"]),

            # 结束符 token id。generate 碰到它就停。
            # config.json / generation_config 里 eos_token_id=151643（本模型 bos 另有 151646）。
            end_token=int(cfg.get("eos_token_id", 151643)),
        )

        # ================================================================
        # 步骤 2：调 C API Create，让 C++ 按 Meta「盖空房子」
        #
        # Create 会做的事（C++ 实现好之后）：
        #   - new 一个 LlaisysQwen2Model 对象
        #   - 按 Meta 给每一块权重 tensorAllocate（形状对、dtype 对，数值还是垃圾）
        #   - 按 nlayer / maxseq / nkvh / dh 预留 KV cache
        #   - 返回一个不透明句柄（void*），Python 只存着，不拆开看内部
        #
        # device / device_ids / ndevice：
        #   单卡 CPU 时就是 device=CPU, ids=[0], ndevice=1。
        #   以后多卡再扩 ids 数组。
        #
        # byref(meta)：把 Python 里的 Structure 地址交给 C，等价于传指针。
        # ================================================================
        device_ids = (c_int * 1)(0)  # C 数组 int[1] = {0}
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta),
            device,
            device_ids,
            1,  # ndevice：只用 1 个设备
        )
        if not self._model:
            raise RuntimeError("llaisysQwen2ModelCreate returned NULL")
        # 留一份给 generate / debug 用，避免反复读 C 侧。
        self._meta = meta
        self._device = device

        # ================================================================
        # 步骤 3：拿 Weights「门牌号表」，再按名字把数值搬进去
        #
        # llaisysQwen2ModelWeights(model) 返回的是：
        #   POINTER(LlaisysQwen2Weights)
        # 也就是「指向 Weights 结构体的指针」。
        # .contents 解引用一次，得到 Python 能用 .字段 访问的那张表。
        #
        # 表里两类字段（见 include/llaisys/models/qwen2.h）：
        #   A. 单个 tensor：
        #        in_embed / out_embed / out_norm_w
        #   B. 长度为 nlayer 的指针数组（每层一块）：
        #        attn_q_w[i], mlp_gate_w[i], ...
        #
        # 这一步只是「拿到门牌」——tensor 壳已经在 Create 时分配好了，
        # 现在要做的是把磁盘上的字节拷进去（tensorLoad）。
        # ================================================================
        weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents

        # 大模型常被切成多个 .safetensors 分片；排序后顺序遍历即可。
        # 每个文件里是一组 (名字 → 数组) 的映射。
        #
        # 注意: 权重是 BF16。普通 numpy 不认 bfloat16 dtype，
        #   framework="numpy" 会报 TypeError: data type 'bfloat16' not understood。
        # 所以用 PyTorch 读出来，再把底层字节指针交给 tensorLoad。
        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as data:
                for name in data.keys():
                    # name 例: "model.layers.0.self_attn.q_proj.weight"
                    # contiguous: 保证一行紧挨一行，tensorLoad 按连续字节 memcpy。
                    weight_cpu = data.get_tensor(name).contiguous()

                    # 名字 → Weights 里对应的那个 llaisysTensor_t 句柄。
                    # 例: "...layers.0...q_proj.weight" → weights.attn_q_w[0]
                    tensor = self._resolve_weight_tensor(weights, name)

                    # tensorLoad(dst, src_ptr)：按 dst 的 numel*elementSize 拷字节。
                    # data_ptr() = torch 张量缓冲区起始地址（BF16 原始位型）。
                    # 必须在 weight_cpu 仍存活时调用，否则指针会悬空。
                    LIB_LLAISYS.tensorLoad(tensor, c_void_p(weight_cpu.data_ptr()))

        # 走到这里：所有 safetensors 条目都已灌进 C++ 模型。
        # generate / Infer 可以开始用这些权重做前向了。

    def _resolve_weight_tensor(self, weights, name: str):
        """把 safetensors 的字符串名字，映射成 Weights 里的一个 tensor 句柄。

        为什么需要映射？
          HuggingFace / safetensors 用的是「层级路径字符串」：
            model.layers.0.self_attn.q_proj.weight
          我们的 C API 用的是「结构体字段 + 层下标」：
            weights.attn_q_w[0]
          两边指的是同一块矩阵，只是命名体系不同。本函数就是词典。

        返回值：
          一个 llaisysTensor_t（ctypes 里的 void* 句柄），
          可以直接传给 tensorLoad。
        """

        # ---------- 全局权重：整网只有一份，不带 layers.i ----------

        # token id → 向量。形状 [voc, hs]。
        # 推理第一步 embedding(tokens) 就查这张表。
        if name == "model.embed_tokens.weight":
            return weights.in_embed

        # 最后隐藏态 → 词表 logits。形状也是 [voc, hs]（linear 的 W）。
        if name == "lm_head.weight":
            return weights.out_embed

        # 所有层跑完之后的最终 RMSNorm 缩放向量。形状 [hs]。
        # HF 叫 model.norm；C 头文件注释写的是 out_norm_w。
        if name == "model.norm.weight":
            return weights.out_norm_w

        # ---------- 逐层权重：名字里嵌着层号 ----------
        #
        # 正则拆两段：
        #   "model.layers.3.mlp.gate_proj.weight"
        #                ^ ^------------------^
        #                |         suffix
        #                i = 3
        #
        # group(1) → 层下标字符串 "3"
        # group(2) → 层内路径     "mlp.gate_proj.weight"
        m = re.match(r"model\.layers\.(\d+)\.(.+)", name)
        if m is None:
            # 既不是三个全局名，也不匹配 layers 模式 → 这份权重我们不认识。
            raise KeyError(f"unknown weight name: {name}")

        i = int(m.group(1))   # 例: 0 .. 27
        suffix = m.group(2)   # 例: "self_attn.q_proj.weight"

        # 层内路径 → Weights 结构体字段名。
        # 字段本身是「长度为 nlayer 的 tensor 指针数组」；
        # 真正取第 i 层要在后面写 [i]。
        #
        # 对照一张前向数据流，会更好记：
        #
        #   x
        #   ├─ RMSNorm(attn_norm_w)                ← input_layernorm
        #   │    Q = x @ q_w^T + q_b               ← q_proj
        #   │    K = x @ k_w^T + k_b               ← k_proj
        #   │    V = x @ v_w^T + v_b               ← v_proj
        #   │    Q,K ← RoPE
        #   │    y  ← self_attention(Q,K,V)
        #   │    y  ← y @ o_w^T                    ← o_proj（无 bias）
        #   x ← x + y
        #   ├─ RMSNorm(mlp_norm_w)                 ← post_attention_layernorm
        #   │    gate = x @ gate_w^T
        #   │    up   = x @ up_w^T
        #   │    h    = swiglu(gate, up)           ← SiLU(gate) ⊙ up
        #   │    y    = h @ down_w^T
        #   x ← x + y
        #
        # 所以：
        #   input_layernorm          → 进 attention 前的 norm
        #   post_attention_layernorm → 进 MLP 前的 norm（名字带 attention，
        #                              实际管的是 MLP 入口）
        layer_map = {
            "input_layernorm.weight": "attn_norm_w",
            "self_attn.q_proj.weight": "attn_q_w",
            "self_attn.q_proj.bias": "attn_q_b",
            "self_attn.k_proj.weight": "attn_k_w",
            "self_attn.k_proj.bias": "attn_k_b",
            "self_attn.v_proj.weight": "attn_v_w",
            "self_attn.v_proj.bias": "attn_v_b",
            "self_attn.o_proj.weight": "attn_o_w",  # Qwen2 的 o_proj 没有 bias
            "post_attention_layernorm.weight": "mlp_norm_w",
            "mlp.gate_proj.weight": "mlp_gate_w",
            "mlp.up_proj.weight": "mlp_up_w",
            "mlp.down_proj.weight": "mlp_down_w",
        }

        field = layer_map.get(suffix)
        if field is None:
            # 层号解析成功，但后缀不在表里（例如将来多了新字段）。
            raise KeyError(f"unknown layer weight: {name}")

        # getattr(weights, "attn_q_w") → 得到 POINTER(llaisysTensor_t)
        # 也就是 C 里的 llaisysTensor_t *，可以按下标取。
        #
        # [i]：取出第 i 层那个句柄。
        # 例: name=...layers.0...q_proj.weight
        #     → getattr(..., "attn_q_w")[0]
        #     → 和 Create 时分配的第 0 层 Q 权重是同一块内存。
        return getattr(weights, field)[i]

    def __del__(self):
        # Python 对象被回收时，把 C++ 模型一并 Destroy，
        # 释放权重 / KV cache 等堆内存，避免泄漏。
        # getattr 是为了防止 __init__ 中途失败、_model 还没赋值就进 __del__。
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """自回归生成：反复调 C++ Infer，拼出整段 token 序列。

        返回值必须是「prompt + 新生成」，不能只返回新 token。
        原因：test_infer.py 里 HF 的 model.generate 返回的是整段
        outputs[0]；--test 时用 `assert llaisys_tokens == tokens` 比整表。

        采样参数 top_k / top_p / temperature：
          C API 的 Infer 只返回一个 int64 token（作业要求 argmax），
          所以这三个参数在 Python 侧用不上。
          --test 会强制 top_k=1, top_p=1, temperature=1，等价贪心，
          与 C++ 里对 logits 做 argmax 一致。保留形参只为对齐测试调用签名。

        与 C++ Infer / KV cache 的约定（两边必须说同一种话）：
          每次把「当前已有的全部 token」传进去（先是 prompt，再逐渐变长）。
          C++ 内部应维护 cache_len（已经写入 KV cache 的长度）：
            - 第一次：cache_len=0，对 ids[0:n] 做 prefill，写入 K/V，再算 next
            - 之后：只对 ids[cache_len:n] 做 decode（通常每次只多 1 个），
                    追加 K/V，再算 next；不要每次从头重算，否则会极慢
        """
        # 测试里没人传 None；默认 128 只是防护。
        _ = (top_k, top_p, temperature)  # 签名兼容；真正采样在 C++ argmax
        if max_new_tokens is None:
            max_new_tokens = 128

        # ids 从 prompt 起步，每步 append 一个新 token。
        # 例: inputs=[1,2,3] → 第一次 Infer 得到 9 → ids=[1,2,3,9] → …
        ids = list(inputs)
        eos = int(self._meta.end_token)

        for _ in range(max_new_tokens):
            n = len(ids)
            # 把 Python list 拷成 C 连续的 int64 数组，Infer 才能按指针读
            buf = (c_int64 * n)(*ids)

            # Infer：跑前向，返回「下一个 token id」（argmax over vocab）
            next_id = int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, buf, n))

            ids.append(next_id)

            # 生成到 EOS 就停；EOS 本身要留在 ids 里（HF 通常也保留）。
            if next_id == eos:
                break

        return ids
