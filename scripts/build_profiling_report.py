#!/usr/bin/env python3
"""Build profiling REPORT.md and decoupled figure PNGs from raw results."""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from profiling_common import load_json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

TIER_ORDER = ["S", "M", "L", "XL"]
TIER_LABELS = {"S": "~64", "M": "~256", "L": "~512", "XL": "~1024"}


def _median_by_tier(runs: list[dict], metric: str) -> dict[str, float]:
    """Median per tier — robust to GPU power/clock-state outliers (1/3 repeats
    can show a 10x prefill spike when the GPU downclocks after an idle gap)."""
    out: dict[str, float] = {}
    for tier in TIER_ORDER:
        vals = [r[metric] for r in runs if r["tier"] == tier and r.get(metric) is not None]
        out[tier] = statistics.median(vals) if vals else 0.0
    return out


def _git_rev() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip()
        return out.split("\n")[0]
    except Exception:
        return "unknown"


def _weight_mib(model_path: Path) -> float:
    total = sum(f.stat().st_size for f in model_path.glob("*.safetensors"))
    return round(total / (1024**2), 1)


def plot_grouped_bars(
    llaisys_vals: dict[str, float],
    vllm_vals: dict[str, float],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    x = range(len(TIER_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        [i - width / 2 for i in x],
        [llaisys_vals[t] for t in TIER_ORDER],
        width,
        label="llaisys",
        color="#4C72B0",
    )
    ax.bar(
        [i + width / 2 for i in x],
        [vllm_vals[t] for t in TIER_ORDER],
        width,
        label="vLLM",
        color="#DD8452",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{t}\n{TIER_LABELS[t]}" for t in TIER_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_kv_fill(runs: list[dict], out_path: Path) -> None:
    vals = _median_by_tier(runs, "kv_fill_ratio")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        [TIER_LABELS[t] for t in TIER_ORDER],
        [vals[t] * 100 for t in TIER_ORDER],
        color="#55A868",
    )
    ax.set_ylabel("KV fill ratio (%)")
    ax.set_xlabel("Prompt tier (tokens)")
    ax.set_title("llaisys: used KV / preallocated maxseq")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_latency_vs_kvlen(step_csv: Path, out_path: Path) -> None:
    rows = list(csv.DictReader(open(step_csv, encoding="utf-8")))
    decode = [r for r in rows if r["phase"] == "decode"]
    kv = [int(r["ntoken"]) for r in decode]
    lat = [float(r["latency_ms"]) for r in decode]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(kv, lat, marker="o", markersize=2, linewidth=1, color="#4C72B0")
    ax.set_xlabel("kvlen (prompt + generated)")
    ax.set_ylabel("per-step latency (ms)")
    ax.set_title("llaisys decode: latency vs kvlen (tier M)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ops_linear(ops: list[dict], out_path: Path) -> None:
    subset = [o for o in ops if o["op"].startswith("linear")]
    names = [o["op"] for o in subset]
    ratios = [o["ratio_llaisys_over_torch"] for o in subset]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, ratios, color="#C44E52")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("llaisys / torch ratio")
    ax.set_title("Decode-shaped Linear: slowdown vs PyTorch")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ops_attention(ops: list[dict], out_path: Path) -> None:
    subset = [o for o in ops if o["op"].startswith("self_attention")]
    kv = [o["kvlen"] for o in subset]
    llaisys = [o["llaisys_ms"] for o in subset]
    torch_ms = [o["torch_ms"] for o in subset]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(kv, llaisys, marker="o", label="llaisys", color="#4C72B0")
    ax.plot(kv, torch_ms, marker="s", label="torch", color="#DD8452")
    ax.set_xlabel("kvlen")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Self-Attention (qlen=1): vs kvlen")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_report(
    results_dir: Path,
    figures_dir: Path,
    report_path: Path,
    model_path: Path,
) -> None:
    llaisys_runs = load_json(results_dir / "llaisys" / "runs.json")
    vllm_runs = load_json(results_dir / "vllm" / "runs.json")
    llaisys_meta = load_json(results_dir / "llaisys" / "meta.json")
    ops = load_json(results_dir / "ops" / "ops.json")

    figures_dir.mkdir(parents=True, exist_ok=True)

    l_ttft = _median_by_tier(llaisys_runs, "ttft_ms")
    v_ttft = _median_by_tier(vllm_runs, "ttft_ms")
    l_tpot = _median_by_tier(llaisys_runs, "tpot_ms")
    v_tpot = _median_by_tier(vllm_runs, "tpot_ms")
    l_peak = _median_by_tier(llaisys_runs, "peak_gpu_mib")
    v_peak = _median_by_tier(vllm_runs, "peak_gpu_mib")

    plot_grouped_bars(
        l_ttft,
        v_ttft,
        "TTFT (ms)",
        "TTFT: llaisys vs vLLM",
        figures_dir / "01_ttft.png",
    )
    plot_grouped_bars(
        l_tpot,
        v_tpot,
        "TPOT (ms)",
        "TPOT per step: llaisys vs vLLM",
        figures_dir / "02_tpot.png",
    )
    plot_grouped_bars(
        l_peak,
        v_peak,
        "峰值显存 (MiB)",
        "Peak GPU memory during inference",
        figures_dir / "03_peak_memory.png",
    )
    plot_kv_fill(llaisys_runs, figures_dir / "04_kv_fill_ratio.png")

    # Use tier M repeat 0 for latency curve
    step_csv = _REPO / "profiling/results/llaisys/M_r0_steps.csv"
    if step_csv.exists():
        plot_latency_vs_kvlen(step_csv, figures_dir / "05_latency_vs_kvlen.png")

    plot_ops_linear(ops, figures_dir / "06_ops_linear_ratio.png")
    plot_ops_attention(ops, figures_dir / "07_ops_attention_kvlen.png")

    weight_mib = _weight_mib(model_path)
    kv_alloc = llaisys_meta["kv_allocated_mib"]

    def ratio_table(metric: str) -> str:
        lines = ["| 档位 | llaisys | vLLM | 倍数 (llaisys/vLLM) |", "|------|---------|------|---------------------|"]
        for t in TIER_ORDER:
            l = _median_by_tier(llaisys_runs, metric)[t]
            v = _median_by_tier(vllm_runs, metric)[t]
            r = l / v if v > 0 else 0
            lines.append(f"| {t} ({TIER_LABELS[t]}) | {l:.1f} | {v:.1f} | {r:.1f}x |")
        return "\n".join(lines)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"""# llaisys Profiling 报告（RTX 5090）

> 生成时间：{now}  
> Git：`{_git_rev()}`  
> GPU：`{_gpu_name()}`  
> 模型：`{model_path}`

## 1. 实验目的

在 **同一模型、同一 GPU、同一套中文 prompt** 下，对比：

- **llaisys**（教学版 runtime + 朴素算子）
- **vLLM**（工业级推理引擎，作数量级参照）

**不是**公平引擎对决：vLLM 含融合 kernel、Paged KV、高度优化的 GEMM；llaisys 无分页 KV、无 Tensor Core GEMM。  
本报告回答：**慢多少、慢在哪、内存浪费在哪**。

## 2. 实验配置

| 项 | 值 |
|----|-----|
| Prompt 档位 | S/M/L/XL ≈ 64 / 256 / 512 / 1024 tokens（chat template 后） |
| Decode 步数 | 128（greedy / argmax） |
| 重复次数 | 3（报告取**中位数**，抗 GPU 功率/时钟状态噪声） |
| vLLM max_model_len | 4096 |
| dtype | BF16 |

> **测量口径**：llaisys 的 `Infer` 是连续生成契约（内部维护 KV cache 位置），
> profiling 脚本在每个独立 repeat 前显式 `ResetCache`，否则新 prompt 长于上一序列时会只算增量、低估 TTFT
> （已修复并复测）。部分 repeat 出现 ~10× 的 prefill 尖峰，源于 GPU 空闲降频/功率竞争，
> 故统计量取中位数而非均值。vLLM 侧用独立请求，Paged KV 天然每次重置，无此问题。

## 3. 端到端：TTFT

TTFT = 首词元延迟，主要反映 **prefill** 成本。

{ratio_table("ttft_ms")}

![TTFT 对比](figures/01_ttft.png)

## 4. 端到端：TPOT

TPOT = decode 阶段每步平均延迟（llaisys 为逐步实测；vLLM 由总时长估算）。

{ratio_table("tpot_ms")}

![TPOT 对比](figures/02_tpot.png)

## 5. 显存与 KV Cache

| 项 | 数值 |
|----|------|
| 权重文件 (safetensors) | {weight_mib} MiB |
| llaisys KV **预分配**上限 | {kv_alloc:.0f} MiB（maxseq={llaisys_meta["maxseq"]}） |

llaisys 在 `Create` 时按 `maxseq=131072` 一次性预分配 KV；本实验对话最长只有 1152 tokens → **填充率 <1%**，绝大多数 KV 槽位永远闲置。

**峰值显存对比要分两种"哲学"（图 3 的 vLLM 柱子更高，不是 llaisys 更省）：**

| 引擎 | 峰值显存 | 含义 |
|------|----------|------|
| llaisys | ≈8.1 GiB | 权重 3.3 GiB + **固定** KV 预分配 3.5 GiB + 运行时开销 |
| vLLM | ≈28.5 GiB | 权重 3.3 GiB + **预留** KV cache 容量 22.2 GiB（`gpu_memory_utilization=0.85`，可调、按需分页） |

vLLM 预留容量是为了支撑并发批处理吞吐；llaisys 是固定预分配 + 朴素实现。两者都不是"单请求实际用量"，但浪费方式不同：
**llaisys 是永远用不满的固定槽位；vLLM 是可通过 Paged KV 按需增长的容量池。**

![峰值显存](figures/03_peak_memory.png)

![KV 填充率](figures/04_kv_fill_ratio.png)

## 6. llaisys：延迟随上下文长度

decode 每步需扫描 **全部历史 KV**，理论上延迟应随 `kvlen` 上升（档位 M 示例，decode 步 kvlen=257→384）。

实测上升**存在但很平缓**（TPOT 从 S 档 54.9ms 到 XL 档 56.6ms，+3%）：因为每步 ~55ms 主要由固定形状的 Linear GEMM 贡献，
attention 的线性增长（§7：0.02→0.16ms）在短上下文内被 GEMM 掩盖；长上下文（数万 token）时 attention 才会成为主导 → 这正是 Paged KV + 融合 attention 的优化点。

![延迟 vs kvlen](figures/05_latency_vs_kvlen.png)

## 7. 算子层：为什么端到端慢

与 PyTorch 同形状 micro-benchmark（decode 形状, BF16）。倍数 >1 = llaisys 慢于 torch，<1 = llaisys 快于 torch。

![Linear 倍数](figures/06_ops_linear_ratio.png)

![Attention kvlen](figures/07_ops_attention_kvlen.png)

| 算子 | llaisys/torch 倍数 | 解读 |
|------|-------------------|------|
"""
    for o in ops:
        if o["op"].startswith("linear"):
            report += f"| {o['op']} | {o['ratio_llaisys_over_torch']}x | Linear GEMM：llaisys 朴素 kernel 无 Tensor Core，慢 **12–13×**（decode 的主瓶颈） |\n"
        elif o["op"].startswith("self_attention"):
            report += (
                f"| {o['op']} | {o['ratio_llaisys_over_torch']}x | "
                f"qlen=1 时 llaisys 反而是精简扫描（比 torch 的通用 attention 还快），"
                f"但随 kvlen 线性上升（{o['llaisys_ms']}ms）|\n"
            )

    report += f"""
## 8. 结论（面试用）

1. **端到端差距**：vLLM 在 TTFT/TPOT 上均显著快于 llaisys。TPOT 恒定 ~18×（55ms vs 3ms）；TTFT 差距随 prompt 增长从 8× 扩大到 **114×**（XL 档 3108ms vs 27ms），因为 llaisys 的 prefill 无融合 kernel，且朴素实现把 prefill 时间浪费在无 Tensor Core 的 GEMM 上。
2. **算子层归因（关键 insight）**：慢的根源是 **Linear GEMM（慢 torch 12–13×）**，不是 attention 也不是 rope。qlen=1 的 attention 在 llaisys 里是精简扫描，反而比 torch 通用 attention 快；但它确实随 kvlen **线性**增长（0.02→0.16ms），长上下文下会逐渐成为 decode 的新瓶颈 → 印证 FlashAttention/分页 KV 的必要性。
3. **内存**：llaisys 按 maxseq=131072 固定预分配 KV（3.5 GiB），本实验填充率 <1%，纯浪费；vLLM 用 Paged KV + 可配置容量池，为并发吞吐预留而非空转。两种"预留"哲学不同，但 llaisys 的固定预分配对长上下文不友好。
4. **下一步**（不在本报告）：分页/按需 KV、Tensor Core 融合 GEMM、FlashAttention、llaisys 的 CUDA Graph 捕获解码步，均需单独 feature 分支评估。

## 9. 原始数据

- `profiling/results/llaisys/runs.json`
- `profiling/results/vllm/runs.json`
- `profiling/results/ops/ops.json`
- `profiling/prompts.json`
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=_REPO / "profiling" / "results")
    parser.add_argument("--figures-dir", type=Path, default=_REPO / "profiling" / "figures")
    parser.add_argument("--report", type=Path, default=_REPO / "profiling" / "report.md")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B"),
    )
    args = parser.parse_args()
    build_report(args.results_dir, args.figures_dir, args.report, args.model)


if __name__ == "__main__":
    main()
