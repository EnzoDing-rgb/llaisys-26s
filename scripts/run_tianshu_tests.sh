#!/usr/bin/env bash
# Run LLAISYS HW4 checks on Iluvatar (天数) after a clean GPU state.
set -euo pipefail

export COREX_HOME="${COREX_HOME:-/usr/local/corex}"
export PATH="$COREX_HOME/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$COREX_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/data/models/DeepSeek-R1-Distill-Qwen-1.5B}"
PYTHON="${PYTHON:-/usr/bin/python3}"

cd "$ROOT"

echo "[1/5] GPU probe"
ixsmi | head -20
"$PYTHON" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.ones(8, device="cuda")
torch.cuda.synchronize()
print("torch compute ok", float(x.sum()))
PY

echo "[2/5] rebuild/install (optional if already built)"
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  xmake f --nv-gpu=y -y
  xmake -j"$(nproc)"
  xmake install
  "$PYTHON" -m pip install -e ./python/ --break-system-packages
fi

echo "[3/5] runtime"
"$PYTHON" test/test_runtime.py --device iluvatar

echo "[4/5] ops"
for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
  echo "---- ops/$op ----"
  "$PYTHON" "test/ops/${op}.py" --device iluvatar
done

echo "[5/5] infer"
"$PYTHON" test/test_infer.py --model "$MODEL_DIR" --test --device iluvatar --max_steps 32

echo "ALL GREEN"
