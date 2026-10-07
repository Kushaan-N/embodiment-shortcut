#!/bin/bash
#
# Architecture A training on UMass Unity (§13).
#
# THE SCRIPT IS BYTE-IDENTICAL AT EVERY ESCALATION STAGE (§13.3).  Only the
# array range and the partition list change, and both come from the command
# line -- never edit this file between S1 and S4.
#
#   S1 (1 item, interactive):
#     salloc -p gpu --gres=gpu:1 --time=02:00:00 --cpus-per-task=8 --mem=64G
#     bash unity/train_idm.sbatch --smoke
#   S2 (5 items):    sbatch --array=0-4  -p gpu             unity/train_idm.sbatch
#   S3 (invite preemption, verify resume BY being preempted):
#                    sbatch --array=0-9  -p gpu,gpu-preempt --time=02:00:00 unity/train_idm.sbatch
#   S4 (full):       sbatch --array=0-29 -p gpu,gpu-preempt --time=02:00:00 unity/train_idm.sbatch
#
# Work list: items 0-19 are the four variants x five seeds; items 20-29 are
# the AUGMENTED A-std / A-del seeds that wm/ladder.yaml scores Experiment H
# with (evaluation.metrics.augmented: true).  --array=0-19 leaves them out.
#
# train_idm.py writes a per-epoch resume state (<tag>/resume.pt) and continues
# from it on the next launch, so an item assembled from 2 h gpu-preempt
# windows loses at most one epoch per kill.  The restart cap below only
# guards against a genuine loop (an item that never completes an epoch).
#
#SBATCH --job-name=ogaf-idm
#SBATCH --output=logs/idm_%A_%a.out
#SBATCH --error=logs/idm_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --time=08:00:00
#            ^^^ MANDATORY.  The default is 1 HOUR on every partition and jobs
#                die at exactly 60 minutes with no useful error (§13.2, §14).
#SBATCH --partition=gpu
#SBATCH --constraint=[a100|l40s]
#            ^^^ select hardware by constraint; gpupod-l40s and superpod-a100
#                are restricted and will silently never schedule.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

set -euo pipefail

# --- environment -----------------------------------------------------------
# SLURM spools a COPY of this script to the node, so ${BASH_SOURCE[0]} does not
# point into the repo under sbatch; SLURM_SUBMIT_DIR does (submit from the repo
# root).  The fallbacks cover interactive `bash unity/*.sbatch` runs.
_OGAF_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[[ -f "$_OGAF_REPO/unity/env.sh" ]] || _OGAF_REPO="$PWD"
if [[ ! -f "$_OGAF_REPO/unity/env.sh" ]]; then
  echo "cannot find unity/env.sh under $_OGAF_REPO; submit from the repo root" >&2
  exit 1
fi
cd "$_OGAF_REPO"
source "$_OGAF_REPO/unity/env.sh"

# --- the work list ---------------------------------------------------------
# One line per (variant, seed[, --augment]).  SLURM_ARRAY_TASK_ID indexes it,
# so the same file drives S2's 5 items and S4's full sweep.
# OGAF_IDM_VARIANTS / OGAF_IDM_AUG=0 select another work list (prereg_addendum_A3.md):
#   OGAF_IDM_VARIANTS="A-std A-del" OGAF_IDM_AUG=0 sbatch --array=0-9 ...
read -r -a VARIANTS <<< "${OGAF_IDM_VARIANTS:-A-std A-del A-time A-clip-del}"
SEEDS=(0 1 2 3 4)
WORK=()
for v in "${VARIANTS[@]}"; do for s in "${SEEDS[@]}"; do WORK+=("$v $s"); done; done
if [[ "${OGAF_IDM_AUG:-1}" != "0" ]]; then
  for v in A-std A-del; do for s in "${SEEDS[@]}"; do WORK+=("$v $s --augment"); done; done
fi

if [[ "${1:-}" == "--smoke" ]]; then
  echo "S1 smoke test: preflight, then ONE short training run (isolated output dir)"
  python unity/preflight.py --time 02:00:00 --partition gpu || exit 1
  set -- ${WORK[0]}
  # --out: a 1-epoch smoke model must NEVER land in the production tag, or the
  # array's skip-if-exists would adopt it as A-std seed 0 for D/E/F/G.
  python train_idm.py --variant "$1" --seed "$2" --epochs 1 \
      --num-workers "${SLURM_CPUS_PER_TASK:-8}" --out "$OGAF_CKPT/_smoke"
  exit $?
fi

IDX=${SLURM_ARRAY_TASK_ID:?this script must run as a job array}
if (( IDX >= ${#WORK[@]} )); then
  echo "array index $IDX beyond work list (${#WORK[@]} items); nothing to do"
  exit 0
fi
set -- ${WORK[$IDX]}
VARIANT=$1
SEED=$2
AUG=${3:-}
TAG="${VARIANT}__seed${SEED}__full${AUG:+__aug}"

echo "=== task $IDX: variant=$VARIANT seed=$SEED aug=${AUG:-no} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date

# Restart cap: with per-epoch resume an item needs a handful of windows at
# most; far more means it never completes an epoch here.  Fail loudly.
if (( ${SLURM_RESTART_COUNT:-0} >= 12 )); then
  echo "task $IDX restarted ${SLURM_RESTART_COUNT} times without finishing: it is not" >&2
  echo "making progress between kills.  Check the epoch time vs --time." >&2
  exit 1
fi
if [[ -f "$OGAF_CKPT/$TAG/resume.pt" ]]; then
  echo "resume state present: continuing ${TAG} from its last completed epoch"
fi

# Skip-if-exists: makes the whole array idempotent under --requeue.
# eval.npz is written atomically, so its existence means a complete run.
if [[ -f "$OGAF_CKPT/$TAG/eval.npz" ]]; then
  echo "already complete; skipping"
  exit 0
fi

# Run in the background and wait: bash defers traps while a FOREGROUND command
# runs, so the old `trap ... TERM` around a foreground srun could never fire.
srun python train_idm.py \
    --variant "$VARIANT" \
    --seed "$SEED" \
    --num-workers "${SLURM_CPUS_PER_TASK:-8}" $AUG &
PID=$!
trap 'echo "SIGTERM received; forwarding to the trainer"; kill -TERM $PID 2>/dev/null || true' TERM
RC=0
while kill -0 "$PID" 2>/dev/null; do
  if wait "$PID"; then RC=0; else RC=$?; fi
done
if (( RC != 0 )); then
  echo "trainer exited with status $RC"
  exit "$RC"
fi

# Validate CONTENT, not the exit code (§13.5).
python - "$OGAF_CKPT/$TAG/eval.npz" <<'PY'
import sys, numpy as np
p = sys.argv[1]
z = np.load(p, allow_pickle=False)
assert "INTERACT_abs_err" in z.files, f"{p}: no INTERACT evaluation was written"
e = z["INTERACT_abs_err"]
assert e.size > 0, f"{p}: empty error array"
assert np.isfinite(e).all(), f"{p}: non-finite errors"
assert e.std() > 0, f"{p}: constant errors -- the model or the loader is broken"
print(f"content OK: n={len(e)} mean_mae={e.mean():.5f}")
PY

date
echo "=== task $IDX done ==="
