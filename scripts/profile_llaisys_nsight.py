#!/usr/bin/env python3
"""Minimal llaisys workload for ncu / nsys capture.

Runs ONE short sequence (a ~256-token prefill + a few decode steps) so the
NVIDIA profilers can capture representative kernels (linear_kernel,
self_attention_kernel, rope_kernel, rms_norm_kernel) without huge traces.

Usage:
  ncu  ... python scripts/profile_llaisys_nsight.py
  nsys ... python scripts/profile_llaisys_nsight.py
"""

from __future__ import annotations

import argparse
import sys
from ctypes import c_int64
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "test"))
sys.path.insert(0, str(_REPO / "scripts"))

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from test_utils import llaisys_device
from transformers import AutoTokenizer

from profiling_common import DEFAULT_MODEL

PROMPT = (
    "大语言模型推理通常分为预填充和解码两个阶段。预填充阶段需要并行处理整段用户提示，"
    "计算开销随提示长度增长；解码阶段每次只生成一个新词元，但要反复读取全部历史键值缓存。"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="nvidia")
    parser.add_argument("--decode-steps", type=int, default=3)
    args = parser.parse_args()

    device = llaisys_device(args.device)
    api = llaisys.RuntimeAPI(device)
    if api.get_device_count() == 0:
        raise RuntimeError(f"No {args.device} device found")
    LIB_LLAISYS.llaisysSetContextRuntime(device, 0)
    api.set_device(0)

    print("Loading llaisys model ...", flush=True)
    model = llaisys.models.Qwen2(str(args.model), device)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = tokenizer.encode(rendered)
    print(f"prompt tokens = {len(ids)}", flush=True)

    # Warmup one prefill (outside profiler interest window).
    LIB_LLAISYS.llaisysQwen2ModelResetCache(model._model)
    buf = (c_int64 * len(ids))(*ids)
    LIB_LLAISYS.llaisysQwen2ModelInfer(model._model, buf, len(ids))
    api.device_synchronize()

    # Reset and run the measured sequence: prefill + a few decode steps.
    LIB_LLAISYS.llaisysQwen2ModelResetCache(model._model)
    ids = list(ids)
    n = len(ids)
    for _ in range(args.decode_steps + 1):
        buf = (c_int64 * n)(*ids)
        next_id = int(LIB_LLAISYS.llaisysQwen2ModelInfer(model._model, buf, n))
        api.device_synchronize()
        ids.append(next_id)
        n += 1

    print("nsight workload done", flush=True)


if __name__ == "__main__":
    main()
