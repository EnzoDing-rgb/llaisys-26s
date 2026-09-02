"""Shared helpers for the profiling experiment suite."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B")
PROMPT_TARGETS = {"S": 64, "M": 256, "L": 512, "XL": 1024}
DECODE_STEPS = 128
REPEATS = 3
WARMUP = 2
VLLM_MAX_MODEL_LEN = 4096


@dataclass
class PromptTier:
    name: str
    target_tokens: int
    text: str
    token_count: int


def kv_mib(cache_len: int, nlayer: int, nkvh: int, dh: int, elem_bytes: int = 2) -> float:
    return 2 * nlayer * cache_len * nkvh * dh * elem_bytes / (1024**2)


def kv_allocated_mib(maxseq: int, nlayer: int, nkvh: int, dh: int, elem_bytes: int = 2) -> float:
    return kv_mib(maxseq, nlayer, nkvh, dh, elem_bytes)


def summarize_decode_rows(rows: list[dict]) -> dict[str, float | None]:
    prefill = [r for r in rows if r["phase"] == "prefill"]
    decode = [r for r in rows if r["phase"] == "decode"]
    ttft = prefill[0]["latency_ms"] if prefill else None
    tpot = sum(r["latency_ms"] for r in decode) / len(decode) if decode else None
    total = sum(r["latency_ms"] for r in rows)
    prompt_tokens = prefill[0]["ntoken"] if prefill else 0
    prefill_per_token = (ttft / prompt_tokens) if ttft and prompt_tokens else None
    return {
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "total_ms": total,
        "prefill_per_token_ms": prefill_per_token,
        "decode_steps": len(decode),
        "prompt_tokens": prompt_tokens,
    }


class GpuMemorySampler:
    """Poll GPU used memory (MiB) during a region."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.02):
        self.device_index = device_index
        self.interval_s = interval_s
        self._peak_mib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _physical_index(self) -> int:
        """Map our logical device index to the physical NVML index.

        Under Slurm, CUDA_VISIBLE_DEVICES remaps e.g. physical GPU 3 to
        logical index 0; NVML ignores that env var and enumerates all GPUs,
        so we must translate to read the right card.
        """
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if vis and vis != "NoDevFiles":
            parts = [p.strip() for p in vis.split(",") if p.strip()]
            if self.device_index < len(parts) and parts[self.device_index].isdigit():
                return int(parts[self.device_index])
        return self.device_index

    def _read_mib(self) -> float:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._physical_index())
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / (1024**2)
        except Exception:
            try:
                import torch

                if torch.cuda.is_available():
                    return torch.cuda.memory_allocated(self.device_index) / (1024**2)
            except Exception:
                pass
        return 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._peak_mib = max(self._peak_mib, self._read_mib())
            time.sleep(self.interval_s)

    def __enter__(self) -> "GpuMemorySampler":
        self._peak_mib = self._read_mib()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._peak_mib = max(self._peak_mib, self._read_mib())

    @property
    def peak_mib(self) -> float:
        return round(self._peak_mib, 2)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prompts_to_json(prompts: list[PromptTier], path: Path) -> None:
    write_json(path, [asdict(p) for p in prompts])
