#!/bin/bash
#
# Shared Unity environment (§13).  Sourced by every sbatch script so cache and
# data policy lives in ONE file instead of drifting across three.
#
#   source unity/env.sh
#
# Everything here is `${VAR:-default}`, so anything already exported on the
# command line or in your shell wins.
#
# WHY THIS FILE EXISTS: the OGAF_* data roots were always redirectable, but the
# *caches* were not.  HF_HOME, TORCH_HOME and the pip/uv caches default to
# $HOME, which on Unity is a small quota shared with your group -- a few GB of
# torch wheels and a 1 GB encoder snapshot will fill it and take your
# colleagues down with you.  Every one of them is pointed at the workspace
# below.

# The repo this file lives in.  Reliable under `source` from anywhere: sbatch
# spools only the *.sbatch script to the node, never this file.
_OGAF_ENV_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The workspace (ws_allocate) that holds corpus, checkpoints and caches.  Hard
# error rather than silently falling back to $HOME, which is the failure this
# file exists to prevent.
export OGAF_DATA=${OGAF_DATA:?set OGAF_DATA to your ws_allocate workspace}

# --- interpreter -----------------------------------------------------------
# Every sbatch script calls bare `python`.  Nothing else activates the venv,
# and the module python has neither torch nor mujoco, so without this every
# array task died with ImportError after its GPU was allocated.  OGAF_PYTHON
# overrides; otherwise the repo venv wins when it exists.
if [[ -n "${OGAF_PYTHON:-}" ]]; then
  export PATH="$(dirname "$OGAF_PYTHON"):$PATH"
elif [[ -x "$_OGAF_ENV_REPO/.venv/bin/python" ]]; then
  export PATH="$_OGAF_ENV_REPO/.venv/bin:$PATH"
fi

# --- data roots (§13.1) ----------------------------------------------------
export OGAF_RESULTS=${OGAF_RESULTS:-$OGAF_DATA/results}
export OGAF_CKPT=${OGAF_CKPT:-$OGAF_DATA/checkpoints}
export OGAF_CORPUS=${OGAF_CORPUS:-$OGAF_DATA/corpus}
export OGAF_CLIPS=${OGAF_CLIPS:-$OGAF_DATA/clips}

# --- committed evidence (§0.3) ---------------------------------------------
# The thresholds were MEASURED ONCE (Experiment A) and every later gate was
# certified against that file.  config.py resolves THRESHOLDS_PATH under
# OGAF_RESULTS -- which now points at an (initially empty) workspace -- so
# every script raised ThresholdsMissing, and the error's suggested fix
# (re-run exp_a) would silently re-measure the thresholds Gates B/C were
# evaluated against.  Pin them to the committed file.  The workspace results
# tree is seeded from the committed gate records below (no-clobber) so
# analyze.py sees Exp 0/A/B/C on Unity; copy NEW results back to the repo
# (results/*/results.json is versioned) before the workspace expires.
export OGAF_THRESHOLDS=${OGAF_THRESHOLDS:-$_OGAF_ENV_REPO/results/exp_a/thresholds.json}

# --- caches: NEVER $HOME ---------------------------------------------------
# HF_HOME covers the hub snapshot cache at $HF_HOME/hub.  It must already
# contain the Gate B encoder: every job below runs HF_HUB_OFFLINE=1, so a cache
# miss is a hard failure at model load -- after the job has queued and been
# allocated a GPU.  `unity/preflight.py` checks this before you spend that.
#
# NOTE: /datasets/ai/dinov2 (Unity's read-only mirror) is NOT a usable default.
# Gate B selected DINOv3; DINOv2 was measured and rejected at 0.55 sigma
# against a 3 sigma threshold (config.GATE_B_EVIDENCE).  Pre-stage DINOv3 here.
export OGAF_CACHE=${OGAF_CACHE:-$OGAF_DATA/cache}
export HF_HOME=${HF_HOME:-$OGAF_CACHE/huggingface}
export TORCH_HOME=${TORCH_HOME:-$OGAF_CACHE/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$OGAF_CACHE/xdg}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$OGAF_CACHE/pip}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$OGAF_CACHE/uv}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$OGAF_CACHE/triton}

# Compute nodes have no outbound network; make that explicit rather than
# discovering it as a timeout mid-training.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# --- rendering -------------------------------------------------------------
# MUST be set before the first `import mujoco`.  Headless GL silently falls
# back to osmesa, which works but is ~10x slower and would corrupt the S2
# wall-clock estimate and therefore the S4 --time request (§13.4).
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

# --- threading -------------------------------------------------------------
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# Data roots must exist and be writable -- fail loudly if not.  logs/ must
# exist in the SUBMIT dir before sbatch runs (#SBATCH --output=logs/...), so
# create it in the repo too, not only under the job's cwd.
mkdir -p "$OGAF_RESULTS" "$OGAF_CKPT" logs "$_OGAF_ENV_REPO/logs"
if [[ -d "$_OGAF_ENV_REPO/results" ]]; then
  cp -rn "$_OGAF_ENV_REPO/results/." "$OGAF_RESULTS/" 2>/dev/null || true
fi

# Caches are different: HF_HOME may legitimately point at a read-only
# pre-staged mirror, and under `set -e` a failed mkdir would kill the job
# before it started.  Create what is missing, warn rather than abort.
for _d in "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" \
          "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$TRITON_CACHE_DIR"; do
  [[ -d "$_d" ]] || mkdir -p "$_d" 2>/dev/null || \
    echo "warn: cannot create cache dir $_d (read-only mirror?)" >&2
done
unset _d
