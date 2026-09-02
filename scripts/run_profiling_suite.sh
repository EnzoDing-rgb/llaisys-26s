#!/usr/bin/env bash
# Full profiling suite: prompts -> llaisys -> vllm -> ops -> report
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-/home/lcpu/39112061/models/DeepSeek-R1-Distill-Qwen-1.5B}"
RESULTS="${RESULTS:-profiling/results}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# matplotlib for report figures
python -c "import matplotlib" 2>/dev/null || uv pip install matplotlib --python "$(which python)"

echo "=== [1/6] Sanity ==="
python test/test_runtime.py --device nvidia
python test/test_infer.py --model "$MODEL" --test --device nvidia --max_steps 4

echo "=== [2/6] Generate Chinese prompts ==="
python scripts/gen_profiling_prompts.py --model "$MODEL"

echo "=== [3/6] llaisys E2E ==="
python scripts/profile_suite_llaisys.py --model "$MODEL" --out-dir "$RESULTS/llaisys"

echo "=== [4/6] vLLM E2E ==="
python scripts/profile_suite_vllm.py --model "$MODEL" --out-dir "$RESULTS/vllm"

echo "=== [5/6] Ops micro-benchmark ==="
python scripts/profile_suite_ops.py --device nvidia

echo "=== [6/6] Build REPORT.md ==="
python scripts/build_profiling_report.py --model "$MODEL"

echo ""
echo "Done. See profiling/report.md"
