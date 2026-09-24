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
# One submission generates ONE ladder model (OGAF_WM_MODEL, default
# WM-base-100).  Experiment H needs all five:
#   for m in WM-base-100 WM-base-30 WM-base-10 WM-data-poor WM-physics-corrupted; do
#     OGAF_WM_MODEL=$m sbatch --array=0-9 -p gpu,gpu-preempt unity/wm_generate.sbatch
#   done
# wm_generate.py resolves WM-base-{100,30,10} to the WM-base run's
# checkpoints at ladder.yaml's checkpoint_at steps.
#
# The determinism and two-action divergence checks run at S1, NOT S2 (§13.3):
# action conditioning being unwired is the one bug that invalidates all
# downstream H data, and two generations are enough to detect it.
#
#SBATCH --job-name=ogaf-wmgen
#SBATCH --output=logs/wmgen_%A_%a.out
#SBATCH --error=logs/wmgen_%A_%a.err
#SBATCH --open-mode=append
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

# --- environment -----------------------------------------------------------
# SLURM spools a COPY of this script to the node, so ${BASH_SOURCE[0]} does not
# point into the repo under sbatch; SLURM_SUBMIT_DIR does (submit from the repo
# root).  The fallback covers interactive `bash unity/*.sbatch` runs.
_OGAF_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[[ -f "$_OGAF_REPO/unity/env.sh" ]] || _OGAF_REPO="$PWD"
if [[ ! -f "$_OGAF_REPO/unity/env.sh" ]]; then
  echo "cannot find unity/env.sh under $_OGAF_REPO; submit from the repo root" >&2
  exit 1
fi
cd "$_OGAF_REPO"
source "$_OGAF_REPO/unity/env.sh"

MODEL=${OGAF_WM_MODEL:-WM-base-100}
GEOMETRY=${OGAF_WM_GEOMETRY:-box}
CONDITION=${OGAF_WM_CONDITION:-INTERACT}   # ABSENT = appearance-gap control (300/model)
N=${OGAF_WM_N:-1500}
FRAMES=${OGAF_WM_FRAMES:-16}     # ladder.yaml transformer.max_frames

case "${1:-}" in
  --preflight)
    exec python unity/preflight.py --time 02:00:00 --partition gpu,gpu-preempt \
        --projected-gb 400 --projected-days 21
    ;;
  --s1)
    echo "S1: one item + determinism + two-action divergence"
    python unity/preflight.py --time 02:00:00 --partition gpu || exit 1
    # OGAF_WM_CKPT=<path> scores an arbitrary checkpoint (e.g. the first
    # ckpt_step*.pt of a run still training) instead of resolving $MODEL.
    python wm/wm_generate.py --model "$MODEL" --geometry "$GEOMETRY" --s1 \
        ${OGAF_WM_CKPT:+--ckpt "$OGAF_WM_CKPT"}
    exit $?
    ;;
esac

IDX=${SLURM_ARRAY_TASK_ID:?run as a job array}
NSHARDS=${SLURM_ARRAY_TASK_COUNT:-1}

echo "=== $MODEL shard $IDX/$NSHARDS on $(hostname) ==="
date

# Background + wait so SLURM's TERM actually reaches the generator (bash
# defers traps while a foreground command runs).  Items are skip-if-exists,
# so a requeued shard resumes where it stopped.
srun python wm/wm_generate.py \
    --model "$MODEL" --geometry "$GEOMETRY" --condition "$CONDITION" --n "$N" \
    --shard "$IDX" --n-shards "$NSHARDS" &
PID=$!
trap 'echo "SIGTERM; forwarding to the generator"; kill -TERM $PID 2>/dev/null || true' TERM
RC=0
while kill -0 "$PID" 2>/dev/null; do
  if wait "$PID"; then RC=0; else RC=$?; fi
done
(( RC == 0 )) || { echo "generator exited with status $RC"; exit "$RC"; }

# Content validation, never exit codes.  --expected-frames: validate.py's
# default is 1, which failed every 16-frame item.  Per-shard manifest name so
# concurrent shards do not overwrite one manifest.json; the final, whole-set
# manifest is written once after the array (see unity/HANDOFF.md).
GEN_DIR="$OGAF_DATA/generated/$MODEL/$GEOMETRY"
[[ "$CONDITION" == "INTERACT" ]] || GEN_DIR="${GEN_DIR}__${CONDITION}"
python unity/validate.py --dir "$GEN_DIR" --expected-frames "$FRAMES" \
    --manifest "$GEN_DIR/manifest_validate_shard$(printf '%03d' "$IDX").json"

date
echo "=== shard $IDX done ==="
