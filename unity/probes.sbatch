#!/bin/bash
#
# Architecture B probes + the §8.4 masking decomposition on Unity (§13).
# ONE sequential job on ONE GPU, deliberately not an array: the expensive
# part is the frozen-encoder (DINOv3 ViT-L, fp32 -- protocol) pass over the
# corpus, which train_idm.py caches per (encoder, geometry, condition,
# horizon, mask, split).  Concurrent array tasks would all race to build the
# same cache files; sequential, the first cell pays and the other 74 fit an
# MLP in seconds each.
#
#   S1  salloc -p gpu --gres=gpu:1 --constraint='[a100|l40s]' --time=02:00:00 \
#             --cpus-per-task=8 --mem=64G
#       bash unity/probes.sbatch --smoke        # one B-std seed, isolated dir
#   S2+ sbatch unity/probes.sbatch              # all 30 probes + 45 masking cells
#
#SBATCH --job-name=ogaf-probes
#SBATCH --output=logs/probes_%j.out
#SBATCH --error=logs/probes_%j.err
#SBATCH --open-mode=append
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=[a100|l40s]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

_OGAF_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[[ -f "$_OGAF_REPO/unity/env.sh" ]] || _OGAF_REPO="$PWD"
if [[ ! -f "$_OGAF_REPO/unity/env.sh" ]]; then
  echo "cannot find unity/env.sh under $_OGAF_REPO; submit from the repo root" >&2
  exit 1
fi
cd "$_OGAF_REPO"
source "$_OGAF_REPO/unity/env.sh"

VARIANTS=(B-std B-del B-time)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

if [[ "${1:-}" == "--smoke" ]]; then
  echo "S1: preflight, then ONE probe into an isolated dir"
  python unity/preflight.py --time 02:00:00 --partition gpu || exit 1
  python train_idm.py --variant B-std --seed 0 --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
      --out "$OGAF_CKPT/_smoke"
  exit $?
fi

echo "=== Architecture B probes on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date
for v in "${VARIANTS[@]}"; do for s in "${SEEDS[@]}"; do
  if [[ -f "$OGAF_CKPT/${v}__seed${s}__full/eval.npz" ]]; then
    echo "skip ${v} seed ${s} (already trained)"; continue
  fi
  python train_idm.py --variant "$v" --seed "$s" --num-workers "${SLURM_CPUS_PER_TASK:-8}"
done; done

echo "=== §8.4 masking decomposition ==="
date
python exp_masking.py --train

# Validate CONTENT: 30 probe runs must have a finite, non-constant eval.
python - <<'PY'
import os, numpy as np
from pathlib import Path
root = Path(os.environ["OGAF_CKPT"])
bad = []
for v in ("B-std", "B-del", "B-time"):
    for s in range(10):
        p = root / f"{v}__seed{s}__full" / "eval.npz"
        if not p.exists():
            bad.append(f"{p}: missing"); continue
        e = np.load(p, allow_pickle=False)["INTERACT_abs_err"]
        if e.size == 0 or not np.isfinite(e).all() or e.std() == 0:
            bad.append(f"{p}: bad content")
assert not bad, "\n".join(bad)
print("content OK: 30 Architecture B probes")
PY
date
echo "=== probes done ==="
