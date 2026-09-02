#!/usr/bin/env python3
"""Profile one vLLM generation with torch.profiler → per-op GPU-time table.

Loads the vLLM engine, runs a short greedy generation (~256-token prompt,
32 decode tokens), and prints the top ops by CUDA time. Industrial standard
for answering "what does one engine step spend its time on".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from profiling_common import DEFAULT_MODEL, VLLM_MAX_MODEL_LEN
from transformers import AutoTokenizer

PROMPT = (
    "大语言模型推理通常分为预填充和解码两个阶段。预填充阶段需要并行处理整段用户提示，"
    "计算开销随提示长度增长；解码阶段每次只生成一个新词元，但要反复读取全部历史键值缓存。"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int, default=VLLM_MAX_MODEL_LEN)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--no-torchprof", action="store_true", help="skip torch.profiler (for nsys-only capture)")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    print("Loading vLLM engine ...", flush=True)
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = tokenizer.encode(rendered)
    print(f"prompt tokens = {len(ids)}", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=args.decode_tokens, ignore_eos=True)
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    # Warmup the engine (graphs already captured at load; run once for allocs).
    llm.generate(ids, sp1)

    if not args.no_torchprof:
        import torch
        from torch.profiler import profile, ProfilerActivity

        print("Profiling generation with torch.profiler ...", flush=True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            llm.generate(ids, sp)

        print("\n=== torch.profiler: top ops by CUDA time ===", flush=True)
        table = prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=25
        )
        print(table, flush=True)

        print("\n=== aggregated by op name (CUDA time) ===", flush=True)
        from collections import defaultdict

        agg: dict[str, float] = defaultdict(float)
        cnt: dict[str, int] = defaultdict(int)
        for evt in prof.key_averages():
            if evt.device_type == 1:  # CUDA
                agg[evt.key] += evt.self_device_time_total
                cnt[evt.key] += evt.count
        total = sum(agg.values()) or 1.0
        for key, t in sorted(agg.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {key:55s} {t/1e3:8.2f} ms ({100*t/total:5.1f}%)  x{cnt[key]}", flush=True)

    print("vllm nsight workload done", flush=True)


if __name__ == "__main__":
    main()
