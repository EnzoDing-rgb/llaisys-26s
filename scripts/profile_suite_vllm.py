#!/usr/bin/env python3
"""Run vLLM end-to-end profiling with aligned prompts and greedy decoding."""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from profiling_common import (
    DECODE_STEPS,
    DEFAULT_MODEL,
    GpuMemorySampler,
    PromptTier,
    VLLM_MAX_MODEL_LEN,
    load_json,
    write_json,
)
from transformers import AutoTokenizer


def tokenize_prompt(model_path: Path, text: str) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(rendered)


def measure_vllm(
    llm,
    prompt_token_ids: list[int],
    decode_steps: int,
) -> dict:
    from vllm import SamplingParams

    import torch

    sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    sp_full = SamplingParams(temperature=0.0, max_tokens=decode_steps, ignore_eos=True)

    torch.cuda.synchronize()
    with GpuMemorySampler() as sampler:
        t0 = time.perf_counter()
        llm.generate(prompt_token_ids, sp1)
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000.0

    torch.cuda.synchronize()
    with GpuMemorySampler() as sampler2:
        t0 = time.perf_counter()
        llm.generate(prompt_token_ids, sp_full)
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000.0
        peak_mib = sampler2.peak_mib

    decode_steps_eff = max(decode_steps - 1, 1)
    tpot_ms = (total_ms - ttft_ms) / decode_steps_eff
    return {
        "ttft_ms": round(ttft_ms, 3),
        "tpot_ms": round(tpot_ms, 3),
        "total_ms": round(total_ms, 3),
        "peak_gpu_mib": peak_mib,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=_REPO / "profiling" / "prompts.json")
    parser.add_argument("--decode-steps", type=int, default=DECODE_STEPS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=VLLM_MAX_MODEL_LEN)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=_REPO / "profiling" / "results" / "vllm")
    args = parser.parse_args()

    prompt_data = [PromptTier(**p) for p in load_json(args.prompts)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM

    print("Loading vLLM engine (may take a few minutes) ...")
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=args.tp_size,
    )

    all_runs: list[dict] = []
    for tier in prompt_data:
        ids = tokenize_prompt(args.model, tier.text)
        for repeat in range(args.repeats):
            metrics = measure_vllm(llm, ids, args.decode_steps)
            record = {
                "engine": "vllm",
                "tier": tier.name,
                "target_tokens": tier.target_tokens,
                "prompt_tokens": len(ids),
                "repeat": repeat,
                "decode_steps": args.decode_steps,
                **metrics,
            }
            all_runs.append(record)
            print(
                f"[vllm] {tier.name}_r{repeat}: TTFT={metrics['ttft_ms']:.1f}ms "
                f"TPOT={metrics['tpot_ms']:.1f}ms"
            )
            gc.collect()

    agg: list[dict] = []
    for tier in prompt_data:
        subset = [r for r in all_runs if r["tier"] == tier.name]
        for key in ("ttft_ms", "tpot_ms", "total_ms", "peak_gpu_mib"):
            vals = [r[key] for r in subset]
            agg.append(
                {
                    "engine": "vllm",
                    "tier": tier.name,
                    "metric": key,
                    "mean": round(statistics.mean(vals), 3),
                    "stdev": round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0,
                }
            )

    write_json(args.out_dir / "runs.json", all_runs)
    write_json(args.out_dir / "aggregate.json", agg)
    write_json(
        args.out_dir / "meta.json",
        {
            "model": str(args.model),
            "max_model_len": args.max_model_len,
            "decode_steps": args.decode_steps,
            "repeats": args.repeats,
        },
    )
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
