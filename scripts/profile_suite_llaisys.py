#!/usr/bin/env python3
"""Run llaisys end-to-end profiling across prompt tiers."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from ctypes import c_int64
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "test"))
sys.path.insert(0, str(_REPO / "scripts"))

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from test_utils import llaisys_device
from transformers import AutoTokenizer

from profiling_common import (
    DECODE_STEPS,
    DEFAULT_MODEL,
    GpuMemorySampler,
    PromptTier,
    kv_allocated_mib,
    kv_mib,
    load_json,
    summarize_decode_rows,
    write_json,
)


def tokenize_prompt(model_path: Path, text: str) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(rendered)


def run_one(
    model,
    api,
    meta,
    ids: list[int],
    decode_steps: int,
) -> tuple[list[dict], float]:
    rows: list[dict] = []
    cache_len_before = 0
    peak_mem = 0.0

    # 关键：llaisys 的 Infer 是连续生成契约，内部维护 cache_len。
    # 每个独立 repeat 前必须重置，否则 ntoken > cache_len 时只会算增量，
    # 导致 TTFT 被低估且 KV cache 混入上一轮的旧序列。
    LIB_LLAISYS.llaisysQwen2ModelResetCache(model._model)

    with GpuMemorySampler() as sampler:
        for step in range(decode_steps + 1):
            ntoken = len(ids)
            n_new = ntoken - cache_len_before
            phase = "prefill" if n_new > 1 else "decode"

            buf = (c_int64 * ntoken)(*ids)
            t0 = time.perf_counter()
            next_id = int(LIB_LLAISYS.llaisysQwen2ModelInfer(model._model, buf, ntoken))
            api.device_synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            rows.append(
                {
                    "step": step,
                    "ntoken": ntoken,
                    "cache_len_before": cache_len_before,
                    "n_new": n_new,
                    "phase": phase,
                    "latency_ms": round(latency_ms, 3),
                    "kv_valid_mib": round(kv_mib(ntoken, meta.nlayer, meta.nkvh, meta.dh), 3),
                }
            )

            if step == decode_steps:
                break
            ids.append(next_id)
            cache_len_before = ntoken
            if next_id == int(meta.end_token):
                break

    peak_mem = sampler.peak_mib
    return rows, peak_mem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="nvidia")
    parser.add_argument("--prompts", type=Path, default=_REPO / "profiling" / "prompts.json")
    parser.add_argument("--decode-steps", type=int, default=DECODE_STEPS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=_REPO / "profiling" / "results" / "llaisys")
    args = parser.parse_args()

    prompt_data = [PromptTier(**p) for p in load_json(args.prompts)]
    device = llaisys_device(args.device)
    api = llaisys.RuntimeAPI(device)
    if api.get_device_count() == 0:
        raise RuntimeError(f"No {args.device} device found")

    LIB_LLAISYS.llaisysSetContextRuntime(device, 0)
    api.set_device(0)

    print("Loading llaisys model ...")
    model = llaisys.models.Qwen2(str(args.model), device)
    meta = model._meta
    kv_max_mib = kv_allocated_mib(meta.maxseq, meta.nlayer, meta.nkvh, meta.dh)

    # Warmup
    warm_ids = tokenize_prompt(args.model, prompt_data[0].text)
    for _ in range(args.warmup):
        buf = (c_int64 * len(warm_ids))(*warm_ids)
        LIB_LLAISYS.llaisysQwen2ModelInfer(model._model, buf, len(warm_ids))
        api.device_synchronize()

    all_runs: list[dict] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for tier in prompt_data:
        for repeat in range(args.repeats):
            ids = tokenize_prompt(args.model, tier.text)
            rows, peak_mib = run_one(model, api, meta, list(ids), args.decode_steps)
            summary = summarize_decode_rows(rows)
            run_id = f"{tier.name}_r{repeat}"
            csv_path = args.out_dir / f"{run_id}_steps.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            last_kv = rows[-1]["kv_valid_mib"]
            fill_ratio = rows[-1]["ntoken"] / meta.maxseq
            record = {
                "engine": "llaisys",
                "tier": tier.name,
                "target_tokens": tier.target_tokens,
                "prompt_tokens": summary["prompt_tokens"],
                "repeat": repeat,
                "ttft_ms": summary["ttft_ms"],
                "tpot_ms": summary["tpot_ms"],
                "total_ms": round(summary["total_ms"], 3),
                "prefill_per_token_ms": summary["prefill_per_token_ms"],
                "decode_steps": summary["decode_steps"],
                "peak_gpu_mib": peak_mib,
                "kv_allocated_mib": round(kv_max_mib, 2),
                "kv_valid_mib_end": last_kv,
                "kv_fill_ratio": round(fill_ratio, 6),
                "steps_csv": str(csv_path.resolve().relative_to(_REPO)),
            }
            all_runs.append(record)
            print(f"[llaisys] {run_id}: TTFT={record['ttft_ms']:.1f}ms TPOT={record['tpot_ms']:.1f}ms")

    # Aggregate per tier
    agg: list[dict] = []
    for tier in prompt_data:
        subset = [r for r in all_runs if r["tier"] == tier.name]
        for key in ("ttft_ms", "tpot_ms", "total_ms", "peak_gpu_mib", "kv_fill_ratio"):
            vals = [r[key] for r in subset if r[key] is not None]
            agg.append(
                {
                    "engine": "llaisys",
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
            "device": args.device,
            "decode_steps": args.decode_steps,
            "repeats": args.repeats,
            "kv_allocated_mib": kv_max_mib,
            "maxseq": meta.maxseq,
            "nlayer": meta.nlayer,
            "nkvh": meta.nkvh,
            "dh": meta.dh,
        },
    )
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
