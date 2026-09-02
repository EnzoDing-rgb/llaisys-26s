#!/usr/bin/env python3
"""Capture decode-shaped op benchmarks into structured JSON."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from test.ops.linear import test_op_linear
from test.ops.self_attention import test_op_self_attention
from test.ops.rope import test_op_rope

from profiling_common import write_json


def _parse_benchmark_output(text: str) -> tuple[float, float]:
    torch_m = re.search(r"Torch time:\s*([\d.]+)\s*ms", text)
    llaisys_m = re.search(r"LLAISYS time:\s*([\d.]+)\s*ms", text)
    if not torch_m or not llaisys_m:
        raise ValueError(f"Failed to parse benchmark output:\n{text}")
    return float(torch_m.group(1)), float(llaisys_m.group(1))


def _bench(fn) -> dict:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    torch_ms, llaisys_ms = _parse_benchmark_output(buf.getvalue())
    ratio = llaisys_ms / torch_ms if torch_ms > 0 else None
    return {
        "torch_ms": round(torch_ms, 4),
        "llaisys_ms": round(llaisys_ms, 4),
        "ratio_llaisys_over_torch": round(ratio, 3) if ratio else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia")
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO / "profiling" / "results" / "ops" / "ops.json",
    )
    args = parser.parse_args()

    dev = args.device
    dtype = "bf16"
    atol, rtol = 1e-2, 1e-2
    profile = True
    results: list[dict] = []

    cases = [
        ("linear_q", lambda: test_op_linear((1, 1536), (1, 1536), (1536, 1536), True, dtype, atol, rtol, dev, profile)),
        ("linear_k", lambda: test_op_linear((1, 256), (1, 1536), (256, 1536), True, dtype, atol, rtol, dev, profile)),
        ("linear_mlp_gate", lambda: test_op_linear((1, 8960), (1, 1536), (8960, 1536), False, dtype, atol, rtol, dev, profile)),
    ]
    for name, fn in cases:
        row = {"op": name, **_bench(fn)}
        results.append(row)

    for kvlen in (128, 512, 1024, 2048):
        name = f"self_attention_kv{kvlen}"
        fn = lambda kv=kvlen: test_op_self_attention(1, kv, 12, 2, 128, dtype, atol, rtol, dev, profile)
        row = {"op": name, "kvlen": kvlen, **_bench(fn)}
        results.append(row)

    row = {
        "op": "rope",
        **_bench(lambda: test_op_rope((1, 12, 128), (511, 512), dtype, atol, rtol, dev, profile)),
    }
    results.append(row)

    write_json(args.out, results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
