#!/usr/bin/env bash
# One-shot profiling on RTX 5090. Run from repo root after build + pip install.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-/home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B}"
OUT_DIR="${OUT_DIR:-profiling/5090_$(date +%Y%m%d_%H%M%S)}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[1/4] Sanity: test_runtime"
python test/test_runtime.py --device nvidia

echo "[2/4] Sanity: test_infer --test (short)"
python test/test_infer.py --model "$MODEL" --test --device nvidia --max_steps 8

echo "[3/4] End-to-end: TTFT / TPOT"
python scripts/profile_infer.py \
  --model "$MODEL" \
  --device nvidia \
  --decode-steps 64 \
  --out "$OUT_DIR/infer"

echo "[4/4] Ops: decode-shaped micro-benchmarks"
python scripts/profile_ops_decode.py --device nvidia | tee "$OUT_DIR/ops_decode.txt"

echo ""
echo "Done. Results under $OUT_DIR/"
echo "  infer: ${OUT_DIR}/infer.csv  ${OUT_DIR}/infer.json"
echo "  ops:   ${OUT_DIR}/ops_decode.txt"
