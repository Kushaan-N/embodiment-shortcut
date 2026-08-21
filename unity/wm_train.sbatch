#!/bin/bash
#
# Experiment H world-model training on UMass Unity (§13).
#
# BUILT TO BE KILLED.  The job trains until a wall-clock budget, checkpoints,
# and exits 0.  SLURM then requeues it and it resumes from the checkpoint.  A
# full 150k-step run is therefore assembled from however many short windows the
# partition will give you -- 2 hours on gpu-preempt, 8 hours, or one long block
# on `gpu`; the script does not care which.
#
# The script is BYTE-IDENTICAL at every escalation stage (§13.3); only the
# partition and array range change, from the command line.
#
#   S0  python unity/preflight.py --time 02:00:00 --partition gpu
#   S1  salloc ... ; bash unity/wm_train.sbatch --smoke
#   S2  sbatch --array=0    -p gpu             unity/wm_train.sbatch
#   S3  sbatch --array=0-2  -p gpu,gpu-preempt unity/wm_train.sbatch   # get preempted
#   S4  sbatch --array=0-2  -p gpu,gpu-preempt unity/wm_train.sbatch
#
#SBATCH --job-name=ogaf-wm
#SBATCH --output=logs/wm_%A_%a.out
#SBATCH --error=logs/wm_%A_%a.err
#SBATCH --time=02:00:00
#            ^^^ MANDATORY. The default is 1 HOUR on every partition (§13.2).
#                Kept at 2h so a gpu-preempt kill can never lose more than the
#                time since the last checkpoint.
#SBATCH --partition=gpu
#SBATCH --constraint=[a100|l40s]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#            ^^^ fire USR1 three minutes before the wall clock expires; the
#                trainer catches it, checkpoints, and exits 0.

set -euo pipefail

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OGAF_DATA=${OGAF_DATA:?set OGAF_DATA to your ws_allocate workspace}
export OGAF_RESULTS=${OGAF_RESULTS:-$OGAF_DATA/results}
export OGAF_CKPT=${OGAF_CKPT:-$OGAF_DATA/checkpoints}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

# Stop ~10 min before the SLURM limit so the final checkpoint always lands.
# Read from the ladder so the two cannot drift apart.
export OGAF_MAX_HOURS=${OGAF_MAX_HOURS:-1.8}

mkdir -p logs "$OGAF_CKPT"

MODELS=(WM-base WM-data-poor WM-physics-corrupted)

if [[ "${1:-}" == "--smoke" ]]; then
  echo "S1: budget report, then 200 steps with a forced checkpoint + resume"
  python unity/preflight.py --time 02:00:00 --partition gpu || exit 1
  python wm/wm_train.py --model WM-base --budget-only
  OGAF_MAX_HOURS=0.02 python wm/wm_train.py --model WM-base --steps 400 --ckpt-every 100
  echo "--- relaunching to prove it RESUMES rather than restarting ---"
  python wm/wm_train.py --model WM-base --steps 400 --ckpt-every 100
  exit $?
fi

IDX=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$IDX]}

echo "=== $MODEL on $(hostname), budget ${OGAF_MAX_HOURS}h, attempt ${SLURM_RESTART_COUNT:-0} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date

# Already finished?  Idempotent under --requeue.
if [[ -f "$OGAF_CKPT/wm/$MODEL/final.pt" ]]; then
  echo "$MODEL already complete; nothing to do"
  exit 0
fi

# The trainer handles USR1/TERM itself; run it in the foreground so the signal
# reaches it.  `srun` would forward the signal to the step, which is what we
# want, but the trap below is the belt-and-braces path.
srun --unbuffered python wm/wm_train.py --model "$MODEL" &
PID=$!
trap 'echo "forwarding signal to trainer"; kill -USR1 $PID 2>/dev/null || true' USR1 TERM
wait $PID || true

# If it exited without producing final.pt, it stopped on budget/preemption.
# Requeue ourselves so the next window continues from the checkpoint.
if [[ ! -f "$OGAF_CKPT/wm/$MODEL/final.pt" ]]; then
  echo "$MODEL incomplete; requeueing to resume from the latest checkpoint"
  date
  scontrol requeue "$SLURM_JOB_ID" || \
    echo "could not self-requeue; resubmit the same sbatch to continue"
  exit 0
fi

# Content validation, never exit codes (§13.5).
python - "$OGAF_CKPT/wm/$MODEL/final.pt" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert "state_dict" in ck and ck["state_dict"], f"{sys.argv[1]}: no weights"
n = sum(v.numel() for v in ck["state_dict"].values())
assert n > 1e7, f"{sys.argv[1]}: only {n} params -- wrong model?"
assert ck["step"] >= ck["cfg"]["train"]["steps"], (
    f"final.pt written at step {ck['step']} but {ck['cfg']['train']['steps']} requested")
print(f"content OK: {n/1e6:.1f}M params at step {ck['step']}")
PY

date
echo "=== $MODEL done ==="
