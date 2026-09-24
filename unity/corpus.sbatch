#!/bin/bash
#
# Corpus build on Unity (§13) -- the CPU-heavy simulation step that must NOT
# run on a login node.  One array task per (geometry, shard): 3 x 20 = 60
# items, each 100 tuples x 3 conditions with BOTH clip variants.  Shards are
# atomic and skip-if-exists (clip-complete), so resubmitting a range is safe.
#
# MuJoCo renders through EGL, which needs a GPU device.  Any GPU will do, so
# the constraint steers to the cheap, plentiful ones and leaves A100/L40S for
# training.  (osmesa on the cpu partition works but is ~10x slower and would
# corrupt every wall-clock estimate.)
#
#   S1  salloc -p gpu --gres=gpu:1 --constraint='[2080_ti|1080_ti]' \
#             --time=01:00:00 --cpus-per-task=4 --mem=16G
#       bash unity/corpus.sbatch --smoke              # box shard 0 (real corpus)
#   S2  sbatch --array=0-4  unity/corpus.sbatch        # record the wall clock
#   S4  sbatch --array=0-59 unity/corpus.sbatch        # everything
#   then (CPU, after the array):
#       sbatch --dependency=afterok:<arrayjobid> unity/run.sbatch python idm_data.py
#       sbatch --dependency=afterok:<arrayjobid> unity/run.sbatch python validate_corpus.py
#
# Friction-swept corpus for WM-physics-corrupted (Experiment H): same file,
#   OGAF_CORPUS_ARGS="--friction-mult 3.0 --conditions INTERACT" \
#       sbatch --array=0-59 unity/corpus.sbatch
#
#SBATCH --job-name=ogaf-corpus
#SBATCH --output=logs/corpus_%A_%a.out
#SBATCH --error=logs/corpus_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#SBATCH --constraint=[2080_ti|1080_ti|titan_x|m40|v100|rtx_8000]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --requeue

set -euo pipefail

_OGAF_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[[ -f "$_OGAF_REPO/unity/env.sh" ]] || _OGAF_REPO="$PWD"
if [[ ! -f "$_OGAF_REPO/unity/env.sh" ]]; then
  echo "cannot find unity/env.sh under $_OGAF_REPO; submit from the repo root" >&2
  exit 1
fi
cd "$_OGAF_REPO"
source "$_OGAF_REPO/unity/env.sh"

EXTRA=${OGAF_CORPUS_ARGS:-}
GEOMS=(box sphere cylinder)
WORK=()
for g in "${GEOMS[@]}"; do for s in $(seq 0 19); do WORK+=("$g $s"); done; done

if [[ "${1:-}" == "--smoke" ]]; then
  echo "S1: preflight (EGL render is the check that matters), then ONE shard"
  python unity/preflight.py --time 01:30:00 --partition gpu || exit 1
  set -- ${WORK[0]}
else
  IDX=${SLURM_ARRAY_TASK_ID:?run as a job array (or --smoke)}
  if (( IDX >= ${#WORK[@]} )); then
    echo "array index $IDX beyond work list (${#WORK[@]} items); nothing to do"; exit 0
  fi
  set -- ${WORK[$IDX]}
fi
GEOMETRY=$1
SHARD=$2

echo "=== $GEOMETRY shard $SHARD on $(hostname) ${EXTRA:+(extra: $EXTRA)} ==="
date
python datasets.py --geometry "$GEOMETRY" --shards "$SHARD" --clips $EXTRA

# Validate CONTENT, not the exit code (§13.5): every condition's shard,
# its validators record, and every clip of every variant must exist.
python - "$GEOMETRY" "$SHARD" $EXTRA <<'PY'
import sys, argparse
import numpy as np
import config as C, datasets as ds
ap = argparse.ArgumentParser()
ap.add_argument("geometry"); ap.add_argument("shard", type=int)
ap.add_argument("--friction-mult", type=float, default=1.0)
ap.add_argument("--conditions", nargs="+", default=list(C.CONDITIONS))
a = ap.parse_args()
lo, hi = a.shard * C.SHARD_SIZE, (a.shard + 1) * C.SHARD_SIZE
for c in a.conditions:
    p = ds.shard_path(C.CORPUS_ROOT, a.geometry, c, a.shard, a.friction_mult)
    assert p.exists(), f"{p} missing"
    assert p.with_suffix(".validators.json").exists(), f"{p}: validators.json missing"
    z = np.load(p, allow_pickle=False)
    assert sorted(int(t) for t in z["tuple_index"]) == list(range(lo, hi)), f"{p}: wrong tuples"
    assert ds.clips_complete(C.CLIP_ROOT, a.geometry, c, range(lo, hi), a.friction_mult), \
        f"{p}: clips incomplete under {C.CLIP_ROOT}"
    print(f"content OK: {p.name} n={len(z['tuple_index'])} "
          f"valid={int(z['validators_passed'].sum())} clips={len(C.CLIP_VARIANTS)}x{hi-lo}")
PY
date
echo "=== $GEOMETRY shard $SHARD done ==="
