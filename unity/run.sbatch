#!/bin/bash
#
# Run ONE command under SLURM with the project environment (§13).  This is
# how every CPU step and every one-off GPU step in unity/HANDOFF.md runs --
# nothing computes on a login node.
#
# The header requests a CPU allocation.  Command-line flags override it, so a
# GPU step is the same file with the hardware asked for on the line:
#
#   # CPU (frame stores, corpus gate, analysis):
#   sbatch unity/run.sbatch python idm_data.py
#   sbatch unity/run.sbatch python validate_corpus.py
#
#   # GPU (Exp B, E, F, G, the VAE, exp_h):
#   sbatch -p gpu --gres=gpu:1 --constraint='[a100|l40s]' --time=02:00:00 \
#       unity/run.sbatch python exp_e_confound.py
#
# --time is MANDATORY on Unity (default 1 h); set it per command.
#
#SBATCH --job-name=ogaf-run
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err
#SBATCH --open-mode=append
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

_OGAF_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[[ -f "$_OGAF_REPO/unity/env.sh" ]] || _OGAF_REPO="$PWD"
if [[ ! -f "$_OGAF_REPO/unity/env.sh" ]]; then
  echo "cannot find unity/env.sh under $_OGAF_REPO; submit from the repo root" >&2
  exit 1
fi
cd "$_OGAF_REPO"
source "$_OGAF_REPO/unity/env.sh"

if (( $# == 0 )); then
  echo "usage: sbatch [slurm flags] unity/run.sbatch <command> [args...]" >&2
  exit 2
fi

echo "=== $(hostname) :: $* ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
date
"$@"
RC=$?
date
echo "=== exit $RC ==="
exit "$RC"
