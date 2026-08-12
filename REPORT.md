# LLAISYS 简要报告

说明本仓库的复现流程，并记录各平台复现结果。  
随 Pull Request 提交；PR 描述可仅摘要并链接本文件。

## 平台总览

| 平台 | 硬件 / 环境 | 状态 |
|------|-------------|------|
| Nvidia | LCPU 集群 `lcpu-infra`，RTX 5090（sm_120） | 已通过 |
| 天数（Iluvatar） | COREX 4.4.0，Iluvatar MR-V100 32GB | 已通过 |

---

## 1. Nvidia（本机 / LCPU 集群）

### 环境

| 项 | 值 |
|----|----|
| 登录节点 | `slurm-login`（编译可在此完成） |
| GPU 分区 | `lcpu-infra`（`gj-5090-*`，RTX 5090 32GB） |
| CUDA | `/usr/local/cuda`，nvcc **13.0** |
| 编译配置 | `xmake f --nv-gpu=y` → `ENABLE_NVIDIA_API`，`xmake/nvidia.lua`（`sm_120`） |
| Python | 仓库内 `.venv`，`torch 2.13.0+cu130` + transformers + llaisys |
| 模型 | `/home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B` |

### 复现流程

```bash
cd /home/lcpu/39112061/llaisys-26s
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

# 1) 打开 Nvidia 后端并编译、安装共享库
xmake f --nv-gpu=y -cv
xmake
xmake install

# 2) 申请 GPU 节点（登录节点无卡，跑测必须进计算节点）
srun -p lcpu-infra --gres=gpu:1 --time=00:15:00 --pty bash
cd /home/lcpu/39112061/llaisys-26s
source .venv/bin/activate

# 3) Runtime
python test/test_runtime.py --device nvidia

# 4) 算子（汇总或单测）
python test/test_ops.py --device nvidia
# 亦可：python test/ops/add.py --device nvidia 等

# 5) 推理正确性（与 PyTorch 对照）
python test/test_infer.py \
  --model /home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --test --device nvidia
```

CPU 基线（CI 同源，本机也可复现）：

```bash
python test/test_runtime.py --device cpu
python test/test_ops.py --device cpu
python test/test_infer.py \
  --model /home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --test --device cpu
```

### 复现结果

| 测试 | 命令 | 结果 |
|------|------|------|
| Runtime (nvidia) | `python test/test_runtime.py --device nvidia` | Passed |
| Ops (nvidia) | `python test/test_ops.py --device nvidia` | Passed |
| Infer (nvidia) | `python test/test_infer.py --model ... --test --device nvidia` | Passed |
| Runtime (cpu) | `python test/test_runtime.py --device cpu` | Passed |
| Infer (cpu) | `python test/test_infer.py --model ... --test --device cpu` | Passed |

实现要点（便于对照代码）：

- Runtime：`src/device/nvidia/`（malloc / memcpy / stream 等）
- 算子：`src/ops/*/nvidia/*.cu`，由各 `op.cpp` 在 `LLAISYS_DEVICE_NVIDIA` 分发
- 推理：`src/llaisys/models.cc` 中权重与中间张量落在 Nvidia，跨设备拷贝走 Runtime `memcpy_sync`

---

## 2. 天数（Iluvatar）

### 环境

| 项 | 值 |
|----|----|
| 实例 / 节点 | 当前容器/实例（工作目录 `/root/llaisys-26s`） |
| GPU | **Iluvatar MR-V100**，32GB（`ixsmi`：IX-ML 4.4.0 / Driver 4.4.0 / CUDA Version 10.2） |
| SDK / 驱动 | COREX **`/usr/local/corex`** → `corex-4.4.0`；`clang++` 在 `$COREX_HOME/bin`；工具链识别为 `GPU stack=iluvatar` |
| 编译方式 | 复用 Nvidia API 路径：`xmake f --nv-gpu=y`；`xmake/nvidia.lua` 自动识别 `/usr/local/corex`，配合 `xmake/iluvatar.lua`；设备枚举仍用 `--device nvidia`（LLAISYS 内部名，实际跑天数） |
| Python | 系统 `python3`；`torch 2.7.1+corex.4.4.0`；`pip install -e ./python/` |
| 模型 | `/data/models/DeepSeek-R1-Distill-Qwen-1.5B`（`hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`） |
| Git commit | `c271775` — *Enable Iluvatar COREX builds alongside NVIDIA CUDA.* |
| 前置检查 | `cudaMalloc` 成功；`torch.ones(8, device='cuda')` → `GPU_OK 8.0` |

关键环境变量：

```bash
export COREX_HOME=/usr/local/corex
export PATH="$COREX_HOME/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/tvm:/usr/local/corex/cpp/libtorch/lib:/usr/local/ixdcgm/lib64:/usr/local/corex/lib64:/usr/local/corex/lib:/usr/local/openmpi/lib:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XMAKE_ROOT=y   # root 用户跑 xmake 必需
```

### 复现流程

```bash
git clone https://github.com/EnzoDing-rgb/llaisys-26s.git
cd llaisys-26s
# 确认 commit 含 xmake/iluvatar.lua、COREX 识别、scripts/run_tianshu_tests.sh

# 若无 xmake：官方脚本装到 ~/.local/bin，root 下加 XMAKE_ROOT=y
export COREX_HOME=/usr/local/corex
export PATH="$COREX_HOME/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$COREX_HOME/lib64:$COREX_HOME/lib:${LD_LIBRARY_PATH:-}"
export XMAKE_ROOT=y

# 1) 编译安装
xmake f --nv-gpu=y -y
xmake -j"$(nproc)"
xmake install
python3 -m pip install -e ./python/ --break-system-packages

# 2) 模型（若不存在）
mkdir -p /data/models
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --local-dir /data/models/DeepSeek-R1-Distill-Qwen-1.5B

# 3) 一键验证（推荐）
SKIP_BUILD=1 MODEL_DIR=/data/models/DeepSeek-R1-Distill-Qwen-1.5B \
  bash scripts/run_tianshu_tests.sh

# 或逐步：
python3 test/test_runtime.py --device nvidia
python3 test/ops/add.py --device nvidia
# ... argmax embedding linear rms_norm rope self_attention swiglu
python3 test/test_infer.py \
  --model /data/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --test --device nvidia --max_steps 32
```

### 复现结果

| 测试 | 命令 | 结果 |
|------|------|------|
| Runtime | `python3 test/test_runtime.py --device nvidia` | Passed（Found 1 nvidia devices） |
| Ops add | `python3 test/ops/add.py --device nvidia` | Passed |
| Ops argmax | `python3 test/ops/argmax.py --device nvidia` | Passed |
| Ops embedding | `python3 test/ops/embedding.py --device nvidia` | Passed |
| Ops linear | `python3 test/ops/linear.py --device nvidia` | Passed |
| Ops rms_norm | `python3 test/ops/rms_norm.py --device nvidia` | Passed |
| Ops rope | `python3 test/ops/rope.py --device nvidia` | Passed |
| Ops self_attention | `python3 test/ops/self_attention.py --device nvidia` | Passed |
| Ops swiglu | `python3 test/ops/swiglu.py --device nvidia` | Passed |
| Infer | `python3 test/test_infer.py --model ... --test --device nvidia --max_steps 32` | Passed（token 与 PyTorch 参考一致；参考 ~1.74s，本实现 ~9.56s） |
| 汇总脚本 | `scripts/run_tianshu_tests.sh` | **ALL GREEN** |

### 备注

- `--device nvidia` 是 LLAISYS 内部设备名；实际由 COREX clang 把同一套 `.cu` 编到天数运行。
- xmake 日志应出现：`[llaisys] GPU stack=iluvatar root=/usr/local/corex`。
- 若 `cudaMalloc` / torch CUDA 分配卡在 `before malloc`，优先判容器/GPU 状态问题，不要先改算子代码。
- root 跑 xmake 必须 `XMAKE_ROOT=y`（或 `--root`）。
- 未在本实例单独跑 CPU 基线；GPU 路径已验收。

---

## 3. 提交与 CI

- 代码以 Pull Request 提交至 [wooway777/llaisys-26s](https://github.com/wooway777/llaisys-26s)
- GitHub Actions 覆盖 **CPU** 路径；**GPU 平台结果以本报告为准**
- 本文件：仓库根目录 `REPORT.md`
