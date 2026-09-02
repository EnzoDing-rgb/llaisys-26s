#!/usr/bin/env python3
"""Generate Chinese prompts at target token counts (after chat template)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from profiling_common import DEFAULT_MODEL, PROMPT_TARGETS, PromptTier, prompts_to_json

# Base Chinese paragraphs (technical / inference themed) for extension.
_BASE_PARAS = [
    "大语言模型推理通常分为预填充和解码两个阶段。预填充阶段需要并行处理整段用户提示，"
    "计算开销随提示长度增长；解码阶段每次只生成一个新词元，但要反复读取全部历史键值缓存。",
    "在自研芯片与框架对接时，我们需要用剖析工具测量首词元延迟和每词元解码延迟，"
    "并分析线性层、注意力层与内存带宽之间的关系，而不是只做端到端的黑盒对比。",
    "键值缓存的布局会显著影响访存效率。若按序列优先存储，解码时沿历史长度扫描会产生较大步长，"
    "导致合并访存失败；若改为按注意力头优先或分页存储，则更容易匹配硬件的最优数据块大小。",
    "与工业级推理引擎相比，教学版运行时往往缺少分页键值缓存、连续批处理和融合算子，"
    "因此性能差距主要来自算子实现与内存管理策略，而非 Python 胶水代码本身。",
    "在图形处理器上，矩阵乘法的性能高度依赖是否使用张量核心以及分块尺寸是否与片上缓存匹配；"
    "朴素实现虽然正确，但通常只能达到框架内置算子几分之一到几十分之一的吞吐。",
    "评测实验应当解耦不同维度：分别测量提示长度、上下文长度、算子类型与峰值显存，"
    "避免把所有指标画在同一张图里造成误读。",
]


def _encode_chat(tokenizer, text: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(rendered)


def _build_text_for_target(tokenizer, target: int) -> tuple[str, int]:
    text = "请用简洁的中文回答以下问题：在推理系统中，为什么解码阶段常常成为瓶颈？\n\n"
    text += _BASE_PARAS[0]
    ids = _encode_chat(tokenizer, text)
    idx = 1
    while len(ids) < target and idx < 200:
        text += _BASE_PARAS[idx % len(_BASE_PARAS)]
        ids = _encode_chat(tokenizer, text)
        idx += 1
    if len(ids) < target:
        pad = " 此外，需要结合硬件峰值带宽、算子融合和缓存命中率来定位瓶颈。"
        while len(ids) < target:
            text += pad
            ids = _encode_chat(tokenizer, text)
    if len(ids) > target + 8:
        # Trim by binary search on suffix length
        lo, hi = 0, len(text)
        best = text
        while lo < hi:
            mid = (lo + hi) // 2
            trial = text[:mid]
            n = len(_encode_chat(tokenizer, trial))
            if n < target:
                lo = mid + 1
            else:
                best = trial
                hi = mid
        text = best
        ids = _encode_chat(tokenizer, text)
    return text, len(ids)


def generate_prompts(model_path: Path) -> list[PromptTier]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    out: list[PromptTier] = []
    for name, target in PROMPT_TARGETS.items():
        text, count = _build_text_for_target(tokenizer, target)
        out.append(PromptTier(name=name, target_tokens=target, text=text, token_count=count))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO / "profiling" / "prompts.json",
    )
    args = parser.parse_args()
    prompts = generate_prompts(args.model)
    prompts_to_json(prompts, args.out)
    for p in prompts:
        print(f"{p.name}: target={p.target_tokens} actual={p.token_count}")


if __name__ == "__main__":
    main()
