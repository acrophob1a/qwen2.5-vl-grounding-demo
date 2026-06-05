#!/usr/bin/env bash
# 一键采集当前实验状态快照（文本留痕，不含大权重）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="records/snapshots/run_${STAMP}.txt"
mkdir -p records/snapshots

{
  echo "=== Snapshot ${STAMP} ==="
  echo
  echo "## Git"
  git rev-parse HEAD 2>/dev/null || echo "no git"
  git status -sb 2>/dev/null || true
  echo
  echo "## Environment"
  python3 -c "import torch; print('torch', torch.__version__)" 2>/dev/null || true
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  echo
  echo "## Dataset"
  wc -l datasets/Grounding-ToyData/toy_data.annotations.tsv 2>/dev/null || true
  echo
  echo "## Latest checkpoint"
  ls -dt work_dirs/grounding-sft-0/checkpoint-* 2>/dev/null | head -1 || echo "none"
  if [ -f work_dirs/grounding-sft-0/trainer_state.json ]; then
    python3 - <<'PY'
import json
from pathlib import Path
p = Path("work_dirs/grounding-sft-0/trainer_state.json")
d = json.loads(p.read_text())
print("global_step:", d.get("global_step"))
print("epoch:", d.get("epoch"))
log = [x for x in d.get("log_history", []) if "train_loss" in x]
if log:
    print("train_loss:", log[-1].get("train_loss"))
PY
  fi
} > "$OUT"

echo "Wrote $OUT"
