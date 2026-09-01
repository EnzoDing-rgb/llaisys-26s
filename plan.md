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

### JD 与本 plan 的对应

| JD 关键词 | 对应步骤 |
|-----------|----------|
| vLLM / 推理引擎对接 | 第 1 步、第 6 步 |
| Runtime + 算子 + C/C++ | 第 2 步、第 3 步 |
| Profiling | 第 4 步、第 5 步 |
| 自研芯片 / CUDA | 第 1 步、第 8 步 |
| 分布式并行 | 第 7 步（CS336 Systems） |
| Triton / 编译器 | 第 8 步顺带 |

---

# llaisys 学习与面试计划

> **岗位核心**：推理引擎与自研芯片栈对接——上层调度保留，下层 runtime + 算子替换。  
> **硬件节奏**：第一轮在 RTX 5090 上跑通；第二轮补天数 Iluvatar（第 8 步）。**当前机器仅 5090。**

## Profiling 作战清单（~半天，5090）

**原则**：先剖自己的 backend（llaisys），再用 vLLM / InfiniLM 作**同指标参照**（不是端到端公平对决）。

### 测什么（三层）

| 层 | 测什么 | 工具 / 脚本 | 产出 |
|----|--------|-------------|------|
| **L3 端到端** | TTFT（prefill 首步）、TPOT（decode 均值）、逐步 `latency_ms` vs `kv_used_mib` | `scripts/profile_infer.py` | `profiling/*.csv` + `*.json` |
| **L2 算子** | decode shape：linear Q/K/MLP；self_attention `kvlen=128/512/1024/2048`；rope | `scripts/profile_ops_decode.py`（内部调 `test/ops/* --profile` 的 `benchmark`） | llaisys vs torch 毫秒数 |
| **参照系**（有余力） | 同模型、同 prompt 的 TTFT/TPOT **数量级** | vLLM：`vllm serve` + 简单计时；或 HF `model.generate`（`test_infer` 已有） | REPORT 一张对照表 |
| **InfiniLM**（本地有才做） | 读 `model_runner` 路径上 TTFT/TPOT；不必先写 tracing | 仓库内 `examples/test_infer.py` 或自带 bench | 「引擎层 vs llaisys runtime 层」一句话 |

**不测**：llaisys 整引擎 vs vLLM 整引擎（缺 Scheduler / Paged KV / fused kernel，比了无信息量）。

### 本机环境（已探测）

| 项目 | 路径 / 状态 |
|------|-------------|
| **llaisys** | `/home/lcpu/39112061/ai-infra/llaisys-26s`（本 workspace） |
| **InfiniLM** | `/home/lcpu/39112061/ai-infra/InfiniLM`（已 clone；**InfiniCore 未装**，`infinilm` 未 pip 进 llaisys venv） |
| **模型** | `/home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B`（默认） |
| **vLLM** | 已装入 llaisys `.venv`：`vllm==0.28.0`（`uv pip install vllm`）；**需在 5090 GPU 节点**上跑 |
| **包管理** | `.venv` 无 `pip` 模块，用 **`uv pip --python .venv/bin/python`** |

InfiniLM 明天若要测：先按 `InfiniLM/README.md` 编 InfiniCore → `pip install -e /home/lcpu/39112061/ai-infra/InfiniLM`，再 `examples/test_infer.py`（自带 `total_time`）。

vLLM 明天对照：5090 上 `python -c "import vllm"` 通过后，用同 prompt 记 TTFT/TPOT（脚本待补 `scripts/profile_vllm.py`）。

### 明天时间怎么切（3～4h）

```text
0:00  环境：xmake --nv-gpu=y 已编、test_infer --test 通过
0:30  bash scripts/run_profile_5090.sh          ← llaisys 主数据
1:30  读 infer.csv：TTFT、TPOT、kvlen 与 latency 趋势
2:00  读 ops_decode.txt：attention 随 kvlen、linear vs torch
2:30  （可选）vLLM 或 HF 一行总耗时 / TTFT
3:00  写 REPORT Profiling 节 + 瓶颈 1～2 条 + InfiniLM 映射一句
```

### 一键命令

```bash
cd /home/lcpu/39112061/ai-infra/llaisys-26s && source .venv/bin/activate
export MODEL=/home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B
bash scripts/run_profile_5090.sh
```

### 预期能得出的结论（填数字后勾选）

- [ ] decode **TPOT** 主因：`self_attention` 读全长 KV（kvlen↑ → ms↑）
- [ ] **linear** naive GEMM 慢于 torch（无 Tensor Core）
- [ ] kernel 内 **`cudaDeviceSynchronize`** / argmax D2H 抬高 TPOT 底噪
- [ ] vLLM/HF 同模型快 **X 倍**（只说数量级 + 分层原因）

---

## 此刻从哪开始

按顺序做。**Profiling 见上文作战清单**；跑通后执行 `scripts/run_profile_5090.sh`。本仓库在第 2～5 步展开；第 6～8 步先列提纲。

```text
第 1 步  三个项目 + JD 标准答
第 2 步  llaisys 架构（本仓库，展开）
第 3 步  5090 跑通 + 读代码（本仓库，展开）
第 4 步  Profiling：TTFT / TPOT / 算子（本仓库，展开）
第 5 步  写入 REPORT + 整理面试话术
第 6 步  vLLM 对照 + InfiniLM 衔接
第 7 步  CS336 分布式并行笔记
第 8 步  天数 Iluvatar + 自研芯片语境
```

---

## 第 1 步：三个项目 + JD 标准答

### 1.1 三个项目的关系

```text
                    ┌─────────────────────────────────────┐
  面试 / JD 常问     │  vLLM（行业参照物）                  │
                    │  Python 调度 + Paged KV + serving    │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
  项目阶段贡献       │  InfiniLM（开源推理引擎）             │
                    │  Scheduler / BlockManager / Server  │
                    └──────────────┬──────────────────────┘
                                   │ compute backend 对接点
                    ┌──────────────▼──────────────────────┐
  现在复习的         │  llaisys（本仓库）                  │
                    │  Runtime API + 算子 + 单序列 Qwen2   │
                    └─────────────────────────────────────┘
```

| | vLLM | InfiniLM | llaisys（本仓库） |
|---|------|----------|-------------------|
| 用途 | 生产引擎心智模型 | 项目 PR、读 Scheduler | 复习、profiling、白板讲调用链 |
| 主体 | Python + CUDA 扩展 | Python + C++ Engine | C++ + C API + ctypes |
| 调度 | Scheduler | Scheduler | `generate` 循环 |
| KV | Paged block | BlockManager | `[maxseq,…]` + `cache_len` |

vLLM / InfiniLM 负责调度与 serving；llaisys 负责 device、算子、单步 forward。JD 要求把手脚接到大脑下面。

### 1.2 vLLM 接自研芯片：留什么、换什么

| 动作 | 层 | 内容 |
|------|-----|------|
| 保留 | 引擎上层（Python） | Scheduler、BlockManager、sampling、HTTP serving |
| 替换 | ModelRunner | attention、linear、rope → 自研 runtime + 算子 |
| 实现 | Device 层 | `LlaisysRuntimeAPI`：malloc / memcpy / stream / sync |
| 验证 | 测试 | `test_ops` + `test_infer`；Nsight / `--profile` |

<div style="font-family: sans-serif; font-size: 14px; line-height: 1.5; max-width: 520px;">

<div style="background:#eef3f8; border-left: 4px solid #5b7c99; padding: 10px 14px; margin-bottom: 8px;">
<b>保留</b> · Scheduler → BlockManager → Server
</div>

<div style="text-align:center; color:#888; margin: 4px 0;">▼</div>

<div style="background:#edf5ee; border-left: 4px solid #6b8f71; padding: 10px 14px; margin-bottom: 8px;">
<b>替换</b> · ModelRunner → attention · linear · rope
</div>

<div style="text-align:center; color:#888; margin: 4px 0;">▼</div>

<div style="background:#f3f0ea; border-left: 4px solid #9a8b6f; padding: 10px 14px; margin-bottom: 8px;">
<b>实现</b> · RuntimeAPI + kernels
</div>

<div style="text-align:center; color:#888; margin: 4px 0;">▼</div>

<div style="background:#f5f5f5; border-left: 4px solid #888; padding: 10px 14px;">
<b>硬件</b> · RTX 5090（第一轮）→ 自研芯片 / 国产 GPU
</div>

</div>

**面试怎么说**：llaisys 练的是对接时要改的那一层（runtime + 算子）；完整引擎还有 Scheduler 和 Paged KV，在 InfiniLM / vLLM 里。

**验收**

- [ ] 三项目各一句话
- [ ] 背出上表四行（保留 / 替换 / 实现 / 验证）

---

## 第 2 步：llaisys 架构（本仓库）

### 2.1 定位

单序列推理 runtime：`libllaisys.so` + Python ctypes，Qwen2 1.5B 与 PyTorch 逐 token 对齐。  
覆盖：device → op → model、KV cache、`cache_len`。  
引擎层能力（Scheduler、Paged KV、多请求 batch）在第 6 步对照 vLLM。

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
  │ L3  模型    models.cc · 28层 · KV · cache   │
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
  └──────────────────────────────────────────┘
```

| 层 | 关键词 |
|----|--------|
| L5 | safetensors、`generate` |
| L4 | C 边界，`__export` |
| L3 | `Infer`、prefill/decode、KV |
| L2 | `op.cpp` 按 device 分发 |
| L1 | `tensorCreate`、`context().runtime()` |
| L0 | `LlaisysRuntimeAPI` |

### 2.3 数据流

```text
config.json + safetensors
  → Meta → Create（权重壳 + KV 预分配）→ Weights（H2D）
  → generate：Infer → next_id → append → 再 Infer
```

- **Prefill**：`past=0`，`n_new=prompt_len` → **TTFT**
- **Decode**：`n_new=1`，`kvlen` 增长 → **TPOT**
- **KV**：每层 `[maxseq, nkvh, dh]`；逻辑长度 = `cache_len`

### 2.4 单层 forward

```text
Embedding → 每层：RMSNorm → QKV → RoPE → 写/读 KV → Attention
  → O proj + residual → RMSNorm → SwiGLU + residual
→ LM head → argmax → cache_len = ntoken
```

### 2.5 KV 账本（1.5B，BF16）

```text
L=28, nkvh=2, dh=128, maxseq=4096
kv_max  ≈ 112 MiB
kv_used(cache_len=512) ≈ 14 MiB
```

### 2.6 Profiling 优先看的算子

```text
self_attention → linear → rope → argmax（D2H / sync）
```

**验收**

- [ ] 画出六层，每层一个关键词
- [ ] 说清 `past` / `n_new` / `cache_len`（prefill vs decode）
- [ ] 手算 `cache_len=512` 的 KV MiB
- [ ] GQA：`nh=12, nkvh=2` 如何读 cache

---

## 第 3 步：5090 跑通 + 读代码（本仓库）

### 3.1 编译与测试

```bash
cd /home/lcpu/39112061/ai-infra/llaisys-26s
source .venv/bin/activate
xmake f --nv-gpu=y -cv && xmake && xmake install && pip install -e ./python/

# GPU 节点：
# srun -p lcpu-infra --gres=gpu:1 --time=00:30:00 --pty bash

python test/test_runtime.py --device nvidia
python test/test_ops.py --device nvidia
python test/test_infer.py \
  --model /home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --test --device nvidia
```

### 3.2 读代码顺序（只盯 shape 与数据流）

1. `python/llaisys/models/qwen2.py` — `generate`
2. `include/llaisys/models/qwen2.h` — API
3. `src/llaisys/models.cc` — `Infer`、KV 读写
4. `src/ops/self_attention/op.cpp` — device 分发
5. `src/device/nvidia/nvidia_runtime_api.cu` — memcpy / sync

### 3.3 环境

| 项 | 值 |
|----|-----|
| 分区 | `lcpu-infra`，RTX 5090，`sm_120` |
| 构建 | `xmake f --nv-gpu=y`，`--device nvidia` |
| 模型 | `/home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B` |

**验收**

- [ ] `test_infer --test` 通过
- [ ] 口述：`safetensors` → Create → Weights → `generate` → `Infer`
- [ ] 口述：28 层循环在算什么

---

## 第 4 步：Profiling（本仓库，5090）

### 4.1 端到端指标

| 指标 | 含义 |
|------|------|
| TTFT | 首次 Infer（prefill）耗时 |
| TPOT | decode 每 token 平均耗时 |
| KV used | 按 `cache_len` 公式 |

记录字段：`step, ntoken, cache_len_before, n_new, phase, latency_ms`  
实现：`scripts/profile_infer.py`（待写）或扩展 `test/test_infer.py`

### 4.2 算子级

```bash
MODEL=/home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B

python test/ops/self_attention.py --device nvidia --profile
python test/ops/linear.py --device nvidia --profile
python test/ops/rope.py --device nvidia --profile

# 端到端（脚本就绪后）
# python scripts/profile_infer.py --device nvidia --model $MODEL ...

# Nsight（有余力）
# nsys profile -o infer_5090 python scripts/profile_infer.py ...
```

### 4.3 关注点

| 主题 | 现象 | 面试表述 |
|------|------|----------|
| Attention | decode 读全长 KV | memory-bound；kvlen↑ → TPOT↑ |
| Linear | naive GEMM | prefill 矩阵更大；Tensor Core 有优化空间 |
| Sync | `deviceSynchronize`、D2H | TPOT 底噪来源之一 |

### 4.4 结果表（填数字）

| 测试 | 结果 |
|------|------|
| test_infer --test | |
| TTFT（prompt ~512） | ms |
| TPOT（decode ~64 tok） | ms |
| KV @ cache_len=512 | MiB |
| 瓶颈结论 | |

**验收**

- [ ] token 序列与 `--test` 一致
- [ ] 报出一个 TTFT、一个 TPOT
- [ ] 指出 decode 主瓶颈（附 ops profile 依据）

---

## 第 5 步：写入 REPORT + 面试话术

### 5.1 REPORT 模板

在 `REPORT.md` 新增：

```markdown
## Profiling（RTX 5090）
### 配置
### 端到端：TTFT / TPOT / KV max & used
### 算子（decode shape）：llaisys vs torch
### 瓶颈结论（1～2 条）
### 与 vLLM 引擎层的分工（一句话，见第 1 步）
```

### 5.2 30 秒话术

> 我实现了 runtime + 算子 + 单序列 Qwen2，KV 用 `cache_len` 区分 prefill/decode，5090 上与 PyTorch 逐 token 对齐。接自研芯片时保留 Scheduler 和 BlockManager，替换 ModelRunner 里的 attention/linear，底层与 RuntimeAPI 同构，用 test_ops/test_infer 做正确性门禁。

### 5.3 考前 30 分钟勾选

- [ ] 第 1 步标准答
- [ ] 第 2 步：prefill/decode、KV 手算、六层
- [ ] 第 4 步：一个 TTFT、一个 TPOT
- [ ] Runtime API 五类：device / stream / malloc / memcpy / sync

**验收**

- [ ] `REPORT.md` Profiling 节已填
- [ ] 话术能脱稿说一遍

---

## 第 6 步：vLLM 对照 + InfiniLM 衔接

llaisys 对应引擎里的 **runtime + 算子层**；vLLM / InfiniLM 用 Python 做 Scheduler，用 pybind / torch.ops 绑 kernel。

| | vLLM / InfiniLM | llaisys |
|---|-----------------|---------|
| 编排 | Scheduler | `generate` |
| KV | Paged block | 连续 `maxseq` |
| 模型执行 | ModelRunner | `models.cc` |

**InfiniLM 概念映射**（开 `InfiniLM` 仓库时）：

| llaisys | InfiniLM |
|---------|----------|
| `cache_len` | `BlockManager` |
| `ModelInfer` | `ModelRunner` |
| `op.cpp` | InfiniCore backend |
| `test_infer --test` | `examples/test_infer.py` |

阅读顺序：`cache_manager.py` → `scheduler.py` → `model_runner.py` → `infer_engine.py`

**验收**

- [ ] 说清 llaisys 在 vLLM 栈里的位置（runtime 层）
- [ ] 列出 InfiniLM 四个先读文件

---

## 第 7 步：CS336 分布式并行

JD 问的 DP / TP / SP / EP 在本地 **CS336 Systems** 笔记里，路径：

`../cs336-assignment2-systems/misc/`

优先读：

- `misc-inference-prefill-decode-batch-tp.md` — 单卡 prefill/decode 算账，再叠 TP
- `misc-llm-parallelism-table.md` — DP/TP/PP/SP/CP/EP 总表

与 llaisys 的衔接：本仓库建立 prefill/decode 与 KV 直觉；CS336 笔记把同一套账接到多卡 collective 上。

**验收**

- [ ] 能说出 DP / TP / EP 各同步什么
- [ ] 能解释：推理 TP 在 decode 阶段通信代价更高（笔记里有数字）

---

## 第 8 步：天数 Iluvatar + 自研芯片语境（第二轮）

第一轮专注 5090。第二轮在云上补天数，REPORT.md §2 已有历史记录。

| | 5090（第一轮） | Iluvatar（第二轮） |
|---|----------------|---------------------|
| 设备 | `--device nvidia` | `--device iluvatar` |
| 环境 | 本地 `lcpu-infra` | COREX 4.4.0 云机 |
| kernel | `src/ops/*/nvidia/*.cu` | 同一套 `.cu` |
| 目标 | profiling 数字 | backend 可移植验证 |

```bash
# 第二轮云上
export COREX_HOME=/usr/local/corex PATH="$COREX_HOME/bin:$PATH" XMAKE_ROOT=y
xmake f --nv-gpu=y -y && xmake -j$(nproc) && xmake install
SKIP_BUILD=1 MODEL_DIR=/data/models/DeepSeek-R1-Distill-Qwen-1.5B \
  bash scripts/run_tianshu_tests.sh
```

**自研芯片（面试概念）**：Tile-Native 分块、高带宽利于 decode KV 读取、LLVM/Triton + 统一 device API。对接步骤同第 1 步四行表。

**验收**

- [ ] 第二轮：`test_infer --test --device iluvatar` 通过
- [ ] 能口述：同 kernel、不同 `LLAISYS_DEVICE_*` 枚举

---

## 本仓库文件索引

| 用途 | 路径 |
|------|------|
| 复现记录 | `REPORT.md` |
| 构建 | `xmake.lua`, `xmake/nvidia.lua` |
| 模型 | `src/llaisys/models.cc` |
| Python | `python/llaisys/models/qwen2.py` |
| 测试 | `test/test_infer.py`, `test/ops/*.py` |
| 天数（第 8 步） | `xmake/iluvatar.lua`, `scripts/run_tianshu_tests.sh` |

---

*第 4 步数字填好后，更新第 4.4 节与 `REPORT.md`。*
