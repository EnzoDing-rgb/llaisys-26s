## 目标岗位 · JD（算苗 AI 框架实习）

### 公司背景

算苗是 3D 堆叠领域的头部企业，全自研。全球率先完成 wafer 级 3D 堆叠算力芯片商业化落地，用全新架构及供应链突破内存墙限制；首款 3D 大模型推理芯片 A4E 于 2026 年 6 月已成功流片，峰值带宽 16 TB/s，帕拉丁仿真性能超越 H200。核心创始团队来自中科院和清华，目前北京 office 接近 200 人规模，预计年底增加至 350+。

### 职位描述

1. 参与深度学习框架 **PyTorch** 与自研芯片软硬件栈对接和性能优化；
2. 参与主流大模型推理引擎（如 **vLLM** 和 **SGLang**）与自研芯片软硬件栈对接，实现大模型的高效推理部署；
3. 参与芯片的**软硬件协同设计**。

### 职位要求

1. 精通 **C/C++** 和 **Python**，有扎实的编程基础、良好的编程风格和工作习惯；
2. 有 AI 领域的基础知识，了解相关算法模型及常见应用领域；
3. 具备主流深度框架或大模型推理引擎开发经验，熟练应用 **Profiling** 技术；
4. 熟悉大模型分布式推理技术（包括数据并行、张量并行、序列并行和专家并行）；
5. 具有良好的团队协作精神，责任心强，能够积极主动地完成相关工作。

### 加分项

1. 熟悉深度学习编译器（如 XLA、TVM、Triton）；
2. 具备大规模模型训练和部署经验；
3. 具备丰富 **CUDA** 并行编程和典型机器学习算子开发经验。

### 本 plan 与 JD 的对应（读 JD 时对照）

| JD 关键词 | 本 plan 落点 |
|-----------|--------------|
| vLLM / 推理引擎对接 | 零节标准答；附录 A |
| Runtime + 算子 + C/C++ | 第二节 llaisys 六层；路径 2 读代码 |
| Profiling | 路径 3；第四节 5090；第五节 REPORT |
| 自研芯片 / CUDA | 零节 Device 层；附录 B；5090 Tensor Core |
| 分布式并行 | 附录 B 概念；项目阶段 InfiniLM |
| Triton / 编译器 | 附录 B bonus，不第一轮深挖 |

---

# llaisys 学习与面试计划

> **怎么用本文**：一次只啃一节。第零节是总图；第一～四节围绕**本仓库 + RTX 5090**；附录备查。  
> **岗位核心**：推理引擎（vLLM 类）与自研芯片栈**对接**——上层调度保留，下层 runtime + 算子替换。  
> **两轮节奏**：**第一轮**在 5090 上把 plan 跑通；**第二轮**再补天数 Iluvatar（见附录 C，与主线解耦）。

---

## 零、总图：三个项目 + JD 标准答（先读）

### 0.1 三个项目各是什么、什么关系

你现在脑子里混在一起的三个东西，其实是**同一条链上的三层**：

```text
                    ┌─────────────────────────────────────┐
  面试 / JD 常问     │  vLLM（行业参照物）                  │
  「生产长什么样」   │  Python 调度 + Paged KV + serving    │
                    └──────────────┬──────────────────────┘
                                   │ 概念同级、工程更完整
                    ┌──────────────▼──────────────────────┐
  项目阶段要贡献     │  InfiniLM（开源推理引擎）             │
  「在引擎里做 PR」  │  Scheduler / BlockManager / Server  │
                    └──────────────┬──────────────────────┘
                                   │ 将来替换的是这一层的 compute backend
                    ┌──────────────▼──────────────────────┐
  你现在复习的       │  llaisys（本仓库，你手搓的）          │
  「下半层肌肉记忆」 │  Runtime API + 算子 + 单序列 Qwen2   │
                    └─────────────────────────────────────┘
```

| | **vLLM** | **InfiniLM** | **llaisys（本仓库）** |
|---|----------|--------------|----------------------|
| **你要用它干什么** | 建立「生产引擎」心智模型；面试对照 | 项目阶段 PR、读真实 Scheduler 代码 | **现在**：复习 + profiling + 白板讲调用链 |
| **谁写的** | 社区 | 开源（你要贡献） | **你** |
| **主体语言** | Python 编排 + CUDA 扩展 | Python 编排 + C++ Engine | C++ 核心 + C API + Python ctypes |
| **有没有 Scheduler** | ✅ | ✅ | ❌（`generate` 简单循环） |
| **KV 怎么管** | Paged block | BlockManager | 连续 `[maxseq,…]` + `cache_len` |
| **你的位置** | 读懂即可，不必 fork | 读 + 做 profiling 类贡献 | **亲手实现** device → op → model |

**一句话**：vLLM / InfiniLM 是**大脑**（调度、batch、KV 池化）；llaisys 是**手脚**（malloc、kernel、单步 forward）。JD 要你会把手脚接到大脑下面。

### 0.2 JD 核心：vLLM 接自研芯片，改哪层、留哪层

这是算苗岗最该背下来的标准答：

| 动作 | 层 | 具体内容 |
|------|-----|----------|
| **保留** | 引擎上层（CPU 侧 Python） | Scheduler、BlockManager、请求状态机、sampling、HTTP serving |
| **替换** | ModelRunner 计算路径 | attention、linear、rope 等 → **自研 runtime + 算子** |
| **实现** | Device 层 | 与 `LlaisysRuntimeAPI` 同构：`malloc` / `free` / `memcpy` / `stream` / `sync` |
| **验证** | 测试门禁 | `test_ops` + `test_infer` → correctness；Nsight / 算子 `--profile` → performance |

<div style="font-family: sans-serif; font-size: 14px; line-height: 1.5; max-width: 520px;">

<div style="background:#eef3f8; border-left: 4px solid #5b7c99; padding: 10px 14px; margin-bottom: 8px;">
<b>保留</b> · vLLM / InfiniLM 侧<br>
Scheduler → BlockManager / Paged KV → Server / 采样
</div>

<div style="text-align:center; color:#888; margin: 4px 0;">▼ 调度结果下发</div>

<div style="background:#edf5ee; border-left: 4px solid #6b8f71; padding: 10px 14px; margin-bottom: 8px;">
<b>替换</b> · llaisys 对应这块<br>
ModelRunner forward → attention · linear · rope · …
</div>

<div style="text-align:center; color:#888; margin: 4px 0;">▼</div>

<div style="background:#f3f0ea; border-left: 4px solid #9a8b6f; padding: 10px 14px; margin-bottom: 8px;">
<b>实现</b> · Device 层<br>
RuntimeAPI + kernels（malloc / memcpy / stream / sync）
</div>

<div style="text-align:center; color:#888; margin: 4px 0;">▼</div>

<div style="background:#f5f5f5; border-left: 4px solid #888; padding: 10px 14px;">
<b>硬件</b> · 第一轮在 RTX 5090 上跑通；自研芯片 / 国产 GPU 同一套接口
</div>

</div>

**面试防踩坑**：

- ❌ 不要说「llaisys 就是 mini vLLM」——缺的是上半层，不是实现错了。
- ❌ 不要担心「C API + ctypes 不符合引擎规范」——bindings 只是手段；**被替换进去的是 runtime + 算子**，不是被否定的部分。
- ✅ 要说：llaisys 练的是**对接时真正要改的那一层**；InfiniLM 是带 Scheduler 的完整引擎，项目阶段在引擎里加 profiling / 优化。

### 0.3 第一轮学习顺序（5090 一条路跑到底）

```text
 0 总图（本节）          ← 三项目 + JD 标准答
 ↓
 二 llaisys 架构        ← 六层 + prefill/decode
 ↓
 三 路径 1→2            ← 自检 + 5090 跑通 test_infer
 ↓
 三 路径 3 + 五          ← profiling → REPORT
 ↓
 四 5090 性能与适配     ← Tensor Core / Nsight；填 TTFT、TPOT
 ↓
 附录 A                 ← 面试前：vLLM 易混淆点
 ↓
 附录 B                 ← 面试前：自研芯片语境（bonus）

 ─── 第二轮（可选，与上面解耦）───
 附录 C                 ← 天数 Iluvatar 云上补测
```

项目阶段再开 InfiniLM 仓库，按附录 B.3 读 `scheduler.py` → `cache_manager.py` → `model_runner.py`。

---

## 一、本仓库节奏（A→E，仅 5090）

```text
 A  架构内化    读懂第二节；能画六层、说清 prefill/decode
 B  复习验证    5090 跑 test_infer --test；口述调用链
 C  Profiling   TTFT / TPOT / KV 账本 + 算子 --profile
 D  5090 深挖   性能数字、瓶颈结论、可选 Nsight
 E  衔接引擎    概念在零节；实现去 InfiniLM 仓库
```

A→B 没过关不要追数字。C→D 是面试弹药。天数 Iluvatar **不在第一轮**，见附录 C。

---

## 二、llaisys 架构（本仓库）

### 2.1 定位

教育向**单序列推理 runtime**：`libllaisys.so` + Python ctypes，Qwen2 1.5B 与 PyTorch 逐 token 对齐。  
**有**：device → op → model + KV + `cache_len`。**没有**：Scheduler、Paged KV、多请求 batch（见零节）。

### 2.2 纵向六层

```text
  ┌──────────────────────────────────────────┐
  │ L5  应用    Qwen2.generate / test_infer   │
  └────────────────────┬─────────────────────┘
                       │
  ┌────────────────────▼─────────────────────┐
  │ L4  C API   llaisys.h / runtime / ops     │
  └────────────────────┬─────────────────────┘
                       │
  ┌────────────────────▼─────────────────────┐
  │ L3  模型    models.cc · 28层 · KV · cache │
  └────────────────────┬─────────────────────┘
                       │
  ┌────────────────────▼─────────────────────┐
  │ L2  算子    embedding / linear / attn …   │
  └────────────────────┬─────────────────────┘
                       │
  ┌────────────────────▼─────────────────────┐
  │ L1  核心    Tensor / Context / Allocator  │
  └────────────────────┬─────────────────────┘
                       │
  ┌────────────────────▼─────────────────────┐
  │ L0  设备    RuntimeAPI → cpu / nvidia     │
  │             （iluvatar 见附录 C，第二轮）   │
  └──────────────────────────────────────────┘
```

| 层 | 关键词 |
|----|--------|
| L5 | safetensors 加载、`generate` 循环 |
| L4 | 稳定 C 边界，`__export` |
| L3 | `Infer`：prefill/decode、`k_cache`/`v_cache` |
| L2 | `op.cpp` 按 device 分发 |
| L1 | `tensorCreate`、`context().runtime()` |
| L0 | `LlaisysRuntimeAPI` 函数表 |

设计要点（各一句）：C API 是边界；算子与模型分离；新芯片 = 新 Runtime 表 + 算子分支；op 前 `setDevice`。

### 2.3 数据流（磁盘 → token）

```text
config.json + safetensors
  → Meta → Create（空壳 + KV 预分配）→ Weights（H2D）
  → generate：Infer → next_id → append → 再 Infer
```

- **KV**：每层 `[maxseq, nkvh, dh]` 预分配；逻辑长度 = `cache_len`
- **Prefill**：`past=0`，`n_new=prompt_len` → 主导 **TTFT**
- **Decode**：`n_new=1`，`kvlen` 涨 → 主导 **TPOT**

### 2.4 单层 forward（Infer 内循环）

```text
Embedding(新 token) → 每层：
  RMSNorm → QKV Linear → RoPE → 写 KV cache → 读全长 KV → Attention
  → O proj + residual → RMSNorm → SwiGLU + residual
→ 末层 RMSNorm → LM head → argmax → cache_len = ntoken
```

### 2.5 算子优先级（profiling 时先盯）

```text
decode 瓶颈候选：self_attention → linear → rope
每步固定：argmax（常伴 D2H / sync）
```

### 2.6 KV 账本（1.5B，BF16）

```text
L=28, nkvh=2, dh=128, maxseq=4096, dtype=2B
kv_max  ≈ 28 × 2 × 4096 × 2 × 128 × 2 ≈ 112 MiB
kv_used ≈ 28 × 2 × cache_len × 2 × 128 × 2
例 cache_len=512 → ≈ 14 MiB
```

---

## 三、学习路径（逐条验收，5090）

### 路径 1 · 地图（不打开代码）

- [ ] 零节：三项目各一句话 + JD 四行标准答
- [ ] 六层各一个关键词
- [ ] `past` / `n_new` / `cache_len`：prefill vs decode
- [ ] 手算 KV：`cache_len=512`
- [ ] GQA：`nh=12, nkvh=2` 怎么读 cache

### 路径 2 · 复习本仓库

```bash
cd /home/lcpu/39112061/ai-infra/llaisys-26s
source .venv/bin/activate
xmake f --nv-gpu=y -cv && xmake && xmake install && pip install -e ./python/

# 需 GPU 节点时：
# srun -p lcpu-infra --gres=gpu:1 --time=00:30:00 --pty bash

python test/test_runtime.py --device nvidia
python test/test_ops.py --device nvidia
python test/test_infer.py \
  --model /home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --test --device nvidia
```

**阅读顺序**（只问 shape 与数据流）：

1. `python/llaisys/models/qwen2.py` — `generate`
2. `include/llaisys/models/qwen2.h` — API
3. `src/llaisys/models.cc` — `Infer` + KV
4. `src/ops/self_attention/op.cpp` — 分发
5. `src/device/nvidia/nvidia_runtime_api.cu` — memcpy/sync

- [ ] `test_infer --test` 通过
- [ ] 能口述 `Infer` 28 层在算什么

### 路径 3 · Profiling

| 指标 | 含义 |
|------|------|
| TTFT | 首次 Infer（prefill） |
| TPOT | decode 平均每 token |
| KV used | 按 `cache_len` 公式 |

记录：`step, ntoken, cache_len_before, n_new, phase, latency_ms`  
落点：`scripts/profile_infer.py`（待实现）或扩展 `test/test_infer.py`

```bash
python test/ops/self_attention.py --device nvidia --profile
python test/ops/linear.py --device nvidia --profile
python test/ops/rope.py --device nvidia --profile
# 可选：nsys profile -o infer python scripts/profile_infer.py ...
```

- [ ] token 序列仍与 `--test` 一致
- [ ] 能报一个 TTFT、一个 TPOT
- [ ] 能指出 decode 瓶颈（attention / sync）

---

## 四、RTX 5090 性能与适配（第一轮主线）

### 4.1 原则

换硬件不动 `models.cc` 逻辑，只动 **RuntimeAPI 表** + **算子 kernel**。  
第一轮只在 5090 上完成 correctness + profiling；国产 GPU 同一套原则，流程见附录 C。

### 4.2 环境

| 项 | 值 |
|----|-----|
| 分区 | `lcpu-infra`，RTX 5090 32GB，`sm_120` |
| CUDA | `/usr/local/cuda`，nvcc 13.0 |
| 构建 | `xmake f --nv-gpu=y` |
| 运行 | `--device nvidia` |
| 模型 | `/home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B` |

### 4.3 命令清单

```bash
MODEL=/home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B

# correctness
python test/test_infer.py --model $MODEL --test --device nvidia

# 算子 profile（decode 形态）
python test/ops/self_attention.py --device nvidia --profile
python test/ops/linear.py --device nvidia --profile
python test/ops/rope.py --device nvidia --profile

# 端到端（profile 脚本就绪后）
# python scripts/profile_infer.py --device nvidia --model $MODEL ...

# 可选 Nsight
# nsys profile -o infer_5090 python scripts/profile_infer.py ...
```

### 4.4 Profiling 关注点

| 主题 | 现状 | 面试怎么说 |
|------|------|------------|
| Attention | decode 读全长 KV | memory-bound；kvlen 涨 TPOT 涨 |
| Linear | naive GEMM | 未吃满 Tensor Core；prefill 更吃亏 |
| Sync | 全局 `deviceSynchronize` 多 | launch + D2H 放大 TPOT 底噪 |
| Tensor Core | 可选优化 | BF16 WMMA / cuBLAS（选做，见附录 B.4） |

### 4.5 结果记录（待填）

| 测试 | 结果 |
|------|------|
| test_infer --test | |
| TTFT（prompt ~512） | ms |
| TPOT（decode ~64 tok） | ms |
| KV @ cache_len=512 | MiB |
| 瓶颈结论 | |

---

## 五、Profiling 写入 REPORT

```markdown
## Profiling（RTX 5090）
### 配置
### 端到端：TTFT / TPOT / KV max & used
### 算子（decode shape）：llaisys vs torch
### 瓶颈结论（1～2 条）
### 与引擎差距（一句话，引用零节）
```

---

## 六、考前自检（30 分钟）

- [ ] 零节标准答能背
- [ ] prefill/decode ↔ `cache_len` / `n_new`
- [ ] KV 手算
- [ ] 调用链：safetensors → Create → Weights → generate → Infer
- [ ] Runtime API 五类：device / stream / malloc / memcpy / sync
- [ ] 一个 TTFT、一个 TPOT（5090）
- [ ] 三项目各一句话

---

## 七、不做的事

- 在 llaisys 里做 Scheduler / PagedAttention
- 未过 correctness 就改 kernel
- 声称「llaisys = vLLM 架构」
- **第一轮**为凑双平台去租云、配 Iluvatar（留第二轮）

---

## 八、本仓库文件索引

| 用途 | 路径 |
|------|------|
| 复现记录 | `REPORT.md` |
| 构建 | `xmake.lua`, `xmake/nvidia.lua` |
| 模型 | `src/llaisys/models.cc` |
| Python | `python/llaisys/models/qwen2.py` |
| 测试 | `test/test_infer.py`, `test/ops/*.py` |
| 天数（第二轮） | `xmake/iluvatar.lua`, `scripts/run_tianshu_tests.sh` |

---

## 附录 A · vLLM 易混淆点（面试前扫）

**vLLM 不是 C API + ctypes 洋葱模型**，而是 Python 大脑 + C++/CUDA 肌肉：

| | vLLM / InfiniLM | llaisys |
|---|-----------------|---------|
| 编排 | Python Scheduler | `generate` for 循环 |
| 绑定 | pybind / torch.ops | ctypes |
| KV | Paged block | 连续 `maxseq` |
| 模型 | ModelRunner 插件 | `models.cc` 手写 |

**llaisys 更像谁**：TensorRT / ONNX Runtime / 芯片 SDK 的 **runtime 层**——正是「接到 vLLM 下面」的那块。

**对接话术（30 秒）**：

> 我手搓了 runtime + 算子 + 单序列 Qwen2，KV 用 `cache_len` 区分 prefill/decode，5090 上与 PyTorch 逐 token 对齐。生产引擎多的是 Scheduler 和 Paged KV；接自研芯片时保留上层调度，替换 ModelRunner 里的 attention/linear，底层实现与 RuntimeAPI 同构，用 test_ops/test_infer 做正确性门禁。

---

## 附录 B · 自研芯片对接（bonus，概念层）

### B.1 算苗 / 自研栈语境

- **Tile-Native**：软件分块与硬件 tile 对齐；GEMM / attention blocking 同一套语言。
- **高带宽内存**：decode 读 KV 常 memory-bound；3D 堆叠高带宽对 TPOT 有利。
- **生态**：LLVM / Triton + 统一 device API；不必 CUDA 二进制兼容。

### B.2 对接步骤（零节标准答展开）

```text
1. malloc/memcpy/stream/sync（对齐 LlaisysRuntimeAPI）
2. 算子：attention / linear / rope … 芯片后端或 Triton
3. 引擎：注册 custom backend，ModelRunner 调你的 op
4. correctness：单算子 → 端到端逐 token
5. performance：TTFT/TPOT + Nsight
```

### B.3 llaisys → InfiniLM 映射（开引擎仓库时用）

| llaisys | InfiniLM |
|---------|----------|
| `cache_len` + 连续 KV | `BlockManager` + `block_table` |
| `ModelInfer` | `ModelRunner` 一步 forward |
| `op.cpp` 分发 | InfiniCore op + backend |
| `generate` | `Scheduler` + server |
| `test_infer --test` | `examples/test_infer.py` |

阅读顺序：`cache_manager.py` → `scheduler.py` → `model_runner.py` → `infer_engine.py`

### B.4 5090 优化方向（选做）

| 方向 | 现状 | 目标 |
|------|------|------|
| GEMM | naive linear | BF16 Tensor Core / cuBLAS |
| Attention | 全长 KV 朴素读 | 分块 / Flash 类 |
| Sync | 全局 sync 多 | stream 化、减 D2H |

---

## 附录 C · 第二轮：天数 Iluvatar（与主线解耦）

> 第一轮**不要求**完成本节。REPORT.md §2 已有历史复现记录；复习时知道「同一套 `.cu` + 不同设备枚举」即可。  
> 第二轮：云上租机 → 按下列清单补跑 → 更新 REPORT 对照一句。

### C.1 和 5090 的关系

```text
  第一轮（5090）                    第二轮（Iluvatar）
  ─────────────────               ─────────────────
  学架构 · profiling · 面试数字      证明 backend 可移植
  --device nvidia                  --device iluvatar
  本地 lcpu-infra                  云上 COREX 环境
```

叙事一句：**兼容层能跑通 ≠ 性能最优**；correctness 门禁与 5090 相同。

### C.2 环境要点

| 项 | 值 |
|----|-----|
| GPU | Iluvatar MR-V100 32GB |
| SDK | COREX 4.4.0，`COREX_HOME=/usr/local/corex` |
| 构建 | 同样 `xmake f --nv-gpu=y`（探测 COREX 后启用 `ILUVATAR`） |
| 运行 | `--device iluvatar`，复用 `src/ops/*/nvidia/*.cu` |

### C.3 云上命令（第二轮用）

```bash
export COREX_HOME=/usr/local/corex
export PATH="$COREX_HOME/bin:$PATH"
export XMAKE_ROOT=y

xmake f --nv-gpu=y -y && xmake -j$(nproc) && xmake install
pip install -e ./python/

SKIP_BUILD=1 MODEL_DIR=/data/models/DeepSeek-R1-Distill-Qwen-1.5B \
  bash scripts/run_tianshu_tests.sh
```

### C.4 第二轮验收

- [ ] `test_infer --test --device iluvatar` 通过
- [ ] 能口述：5090 与 Iluvatar 同 kernel、不同 `LLAISYS_DEVICE_*` 枚举
- [ ] （可选）与 5090 比相对 PyTorch 倍率，不追绝对性能

---

*第一轮 profiling 落地后，更新第四节 4.5 与第五节 REPORT。*
