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
#   S2 (5 items):    sbatch --array=0-4  -p gpu               unity/train_idm.sbatch
#   S3 (50 items):   sbatch --array=0-49 -p gpu,gpu-preempt   unity/train_idm.sbatch
#   S4 (full):       sbatch --array=0-19 -p gpu,gpu-preempt --time=$((3*S2_MEDIAN)) ...
#
#SBATCH --job-name=ogaf-idm
#SBATCH --output=logs/idm_%A_%a.out
#SBATCH --error=logs/idm_%A_%a.err
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
export MUJOCO_GL=egl            # MUST be set before the first `import mujoco`
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-/datasets/ai/dinov2}
export OGAF_DATA=${OGAF_DATA:?set OGAF_DATA to your ws_allocate workspace}
export OGAF_RESULTS=${OGAF_RESULTS:-$OGAF_DATA/results}
export OGAF_CKPT=${OGAF_CKPT:-$OGAF_DATA/checkpoints}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p logs "$OGAF_RESULTS" "$OGAF_CKPT"

# --- clean SIGTERM so --requeue resumes rather than corrupting --------------
trap 'echo "SIGTERM received; exiting cleanly for requeue"; exit 143' TERM

# --- the work list ---------------------------------------------------------
# One line per (variant, seed).  SLURM_ARRAY_TASK_ID indexes it, so the same
# file drives S2's 5 items and S4's full sweep.
VARIANTS=(A-std A-del A-time A-clip-del)
SEEDS=(0 1 2 3 4)
WORK=()
for v in "${VARIANTS[@]}"; do for s in "${SEEDS[@]}"; do WORK+=("$v $s"); done; done

if [[ "${1:-}" == "--smoke" ]]; then
  echo "S1 smoke test: preflight, then ONE short training run"
  python unity/preflight.py --time 02:00:00 --partition gpu || exit 1
  set -- ${WORK[0]}
  python train_idm.py --variant "$1" --seed "$2" --epochs 1 \
      --num-workers "${SLURM_CPUS_PER_TASK:-8}"
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

echo "=== task $IDX: variant=$VARIANT seed=$SEED on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date

# Skip-if-exists: makes the whole array idempotent under --requeue.
if [[ -f "$OGAF_CKPT/${VARIANT}__seed${SEED}__full/eval.npz" ]]; then
  echo "already complete; skipping"
  exit 0
fi

srun python train_idm.py \
    --variant "$VARIANT" \
    --seed "$SEED" \
    --num-workers "${SLURM_CPUS_PER_TASK:-8}"

# Validate CONTENT, not the exit code (§13.5).
python - "$OGAF_CKPT/${VARIANT}__seed${SEED}__full/eval.npz" <<'PY'
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
