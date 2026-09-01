#!/usr/bin/env python3
"""End-to-end infer profiling: TTFT, TPOT, per-step CSV.

Usage (5090):
  source .venv/bin/activate
  python scripts/profile_infer.py \\
    --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \\
    --device nvidia \\
    --decode-steps 64 \\
    --out profiling/infer_5090
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from ctypes import c_int64
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "test"))

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from test_utils import llaisys_device
from transformers import AutoTokenizer


def kv_mib(cache_len: int, nlayer: int, nkvh: int, dh: int, elem_bytes: int = 2) -> float:
    """Per-layer K+V; total = 2 * nlayer * cache_len * nkvh * dh * elem_bytes."""
    return 2 * nlayer * cache_len * nkvh * dh * elem_bytes / (1024**2)


def tokenize_prompt(model_path: Path, prompt: str) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    text = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(text)


def run_profile(
    model_path: Path,
    device_name: str,
    prompt: str,
    decode_steps: int,
    warmup: int,
    out_prefix: Path,
) -> dict:
    device = llaisys_device(device_name)
    api = llaisys.RuntimeAPI(device)
    if api.get_device_count() == 0:
        raise RuntimeError(f"No {device_name} device found")

    LIB_LLAISYS.llaisysSetContextRuntime(device, 0)
    api.set_device(0)

    print(f"Loading model from {model_path} ...")
    model = llaisys.models.Qwen2(str(model_path), device)
    m = model._meta

    ids = tokenize_prompt(model_path, prompt)
    prompt_len = len(ids)
    print(f"Prompt tokens: {prompt_len}")

    # Warmup on the same model object (weights stay loaded).
    for _ in range(warmup):
        warm_ids = list(ids)
        buf = (c_int64 * len(warm_ids))(*warm_ids)
        LIB_LLAISYS.llaisysQwen2ModelInfer(model._model, buf, len(warm_ids))
        api.device_synchronize()
        for _ in range(min(3, decode_steps)):
            buf = (c_int64 * len(warm_ids))(*warm_ids)
            nid = int(LIB_LLAISYS.llaisysQwen2ModelInfer(model._model, buf, len(warm_ids)))
            api.device_synchronize()
            warm_ids.append(nid)

    # Reset KV: ntoken <= cache_len clears cache (see models.cc).
    ids = tokenize_prompt(model_path, prompt)
    rows: list[dict] = []
    cache_len_before = 0

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
                "kv_used_mib": round(kv_mib(ntoken, m.nlayer, m.nkvh, m.dh), 3),
            }
        )

        if step == decode_steps:
            break

        ids.append(next_id)
        cache_len_before = ntoken

        if next_id == int(m.end_token):
            print(f"EOS at step {step}, stopping early.")
            break

    prefill_rows = [r for r in rows if r["phase"] == "prefill"]
    decode_rows = [r for r in rows if r["phase"] == "decode"]

    ttft_ms = prefill_rows[0]["latency_ms"] if prefill_rows else None
    tpot_ms = (
        sum(r["latency_ms"] for r in decode_rows) / len(decode_rows) if decode_rows else None
    )

    summary = {
        "device": device_name,
        "model": str(model_path),
        "prompt_tokens": prompt_len,
        "decode_steps_measured": len(decode_rows),
        "ttft_ms": ttft_ms,
        "tpot_ms": round(tpot_ms, 3) if tpot_ms is not None else None,
        "kv_max_mib": round(kv_mib(m.maxseq, m.nlayer, m.nkvh, m.dh), 2),
        "kv_at_last_step_mib": rows[-1]["kv_used_mib"] if rows else None,
    }

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile llaisys Qwen2 infer (TTFT / TPOT)")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B"),
    )
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia", "iluvatar"])
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--decode-steps", default=64, type=int, help="decode steps after prefill")
    parser.add_argument("--warmup", default=1, type=int)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("profiling/infer_5090"),
        help="Output path prefix (writes .csv and .json)",
    )
    args = parser.parse_args()
    run_profile(
        args.model,
        args.device,
        args.prompt,
        args.decode_steps,
        args.warmup,
        args.out,
    )


if __name__ == "__main__":
    main()
