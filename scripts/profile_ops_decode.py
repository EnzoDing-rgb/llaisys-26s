#!/usr/bin/env python3
"""Decode-shaped op micro-benchmarks (llaisys vs torch).

Shapes match Qwen2-1.5B BF16 decode (qlen=1). Sweeps kvlen for attention.

Usage:
  python scripts/profile_ops_decode.py --device nvidia
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "test"))

from test.ops.linear import test_op_linear
from test.ops.self_attention import test_op_self_attention
from test.ops.rope import test_op_rope


# Qwen2-1.5B: hs=1536, nh=12, nkvh=2, dh=128, di=8960
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia", "iluvatar"])
    parser.add_argument("--warmup", default=10, type=int, help="passed via test_utils.benchmark")
    args = parser.parse_args()

    dtype = "bf16"
    atol, rtol = 1e-2, 1e-2
    dev = args.device

    print(f"=== Decode-shaped ops on {dev} (BF16) ===\n")

    print("--- linear: Q proj (1,1536) @ (1536,1536) ---")
    test_op_linear((1, 1536), (1, 1536), (1536, 1536), True, dtype, atol, rtol, dev, True)

    print("\n--- linear: K proj (1,256) @ (256,1536) ---")
    test_op_linear((1, 256), (1, 1536), (256, 1536), True, dtype, atol, rtol, dev, True)

    print("\n--- linear: MLP gate (1,8960) @ (8960,1536) ---")
    test_op_linear((1, 8960), (1, 1536), (8960, 1536), False, dtype, atol, rtol, dev, True)

    for kvlen in (128, 512, 1024, 2048):
        print(f"\n--- self_attention: qlen=1 kvlen={kvlen} nh=12 nkvh=2 hd=128 ---")
        test_op_self_attention(1, kvlen, 12, 2, 128, dtype, atol, rtol, dev, True)

    print("\n--- rope: (1,12,128) pos=[511] ---")
    test_op_rope((1, 12, 128), (511, 512), dtype, atol, rtol, dev, True)

    print("\n\033[92mOp profiling done.\033[0m")


if __name__ == "__main__":
    main()
