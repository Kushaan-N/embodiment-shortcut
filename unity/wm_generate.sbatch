#!/bin/bash
#
# Experiment H generation on Unity (§13).  Same escalation discipline as
# train_idm.sbatch: the file is byte-identical at every stage.
#
#   S0  bash unity/wm_generate.sbatch --preflight        # login node
#   S1  salloc ... ; bash unity/wm_generate.sbatch --s1  # 1 item + S1 CHECKS
#   S2  sbatch --array=0-4  -p gpu             unity/wm_generate.sbatch
#   S3  sbatch --array=0-49 -p gpu,gpu-preempt unity/wm_generate.sbatch
#   S4  sbatch --array=0-N  -p gpu,gpu-preempt --time=<3x S2 median> ...
#
# The determinism and two-action divergence checks run at S1, NOT S2 (§13.3):
# action conditioning being unwired is the one bug that invalidates all
# downstream H data, and two generations are enough to detect it.
#
#SBATCH --job-name=ogaf-wmgen
#SBATCH --output=logs/wmgen_%A_%a.out
#SBATCH --error=logs/wmgen_%A_%a.err
#SBATCH --time=02:00:00
#            ^^^ MANDATORY, and <= 2h so gpu-preempt kills cannot lose work.
#SBATCH --partition=gpu
#SBATCH --constraint=[a100|l40s]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

set -euo pipefail

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OGAF_DATA=${OGAF_DATA:?set OGAF_DATA to your ws_allocate workspace}
export OGAF_CKPT=${OGAF_CKPT:-$OGAF_DATA/checkpoints}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p logs

trap 'echo "SIGTERM; exiting cleanly for requeue"; exit 143' TERM

MODEL=${OGAF_WM_MODEL:-WM-base-100}
GEOMETRY=${OGAF_WM_GEOMETRY:-box}
N=${OGAF_WM_N:-1500}

case "${1:-}" in
  --preflight)
    exec python unity/preflight.py --time 02:00:00 --partition gpu,gpu-preempt \
        --projected-gb 400 --projected-days 21
    ;;
  --s1)
    echo "S1: one item + determinism + two-action divergence"
    python unity/preflight.py --time 02:00:00 --partition gpu || exit 1
    python wm/wm_generate.py --model "$MODEL" --geometry "$GEOMETRY" --s1
    exit $?
    ;;
esac

IDX=${SLURM_ARRAY_TASK_ID:?run as a job array}
NSHARDS=${SLURM_ARRAY_TASK_COUNT:-1}

echo "=== $MODEL shard $IDX/$NSHARDS on $(hostname) ==="
date

srun python wm/wm_generate.py \
    --model "$MODEL" --geometry "$GEOMETRY" --n "$N" \
    --shard "$IDX" --n-shards "$NSHARDS"

# Content validation, never exit codes.
python unity/validate.py --dir "$OGAF_DATA/generated/$MODEL/$GEOMETRY"

date
echo "=== shard $IDX done ==="
