#!/bin/bash
#
# Submit the whole OG-AF pipeline on Unity as ONE dependency chain, from any
# stage onward.  Every script it submits is skip-if-exists, so re-running a
# stage is idempotent, and a failed gate stops everything after it
# (DependencyNeverSatisfied).  This is the chain unity/HANDOFF.md §2 describes,
# executed instead of typed.
#
#   source <workspace>/activate.sh        # OGAF_DATA, venv, SBATCH_ACCOUNT
#   bash unity/pipeline.sh --from corpus  # stages: corpus idm floors prereg-check e h
#   bash unity/pipeline.sh --from h --dry-run
#
# Stages (in order; --from picks the first one to submit):
#   corpus       60-way corpus array -> frame stores -> corpus gate -> Exp 0 (+subset audit)
#   idm          Architecture B probes + masking; Architecture A (30 items, preemptible)
#   floors       Experiment D floors -> power analysis            (then: write prereg.md)
#   prereg-check prereg_lock.py must PASS before anything below is submitted
#   e            Experiments E, F (generate+score), G -> analyze
#   h            friction x3 corpus + WM clips -> VAE -> WM smoke -> caches -> 3 runs
#                -> S1 gen check -> 5 models x (INTERACT + ABSENT) -> validate -> exp_h -> analyze
#   gen          only the tail of h: generation -> validate -> exp_h -> analyze (models already trained)
set -euo pipefail
: "${OGAF_DATA:?source the workspace activate.sh (or unity/env.sh) first}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source unity/env.sh

FROM=corpus; DRY=0
while (( $# )); do case "$1" in
  --from) FROM=$2; shift 2;; --dry-run) DRY=1; shift;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done
ORDER=(corpus idm floors prereg-check e h gen)
start=-1; for i in "${!ORDER[@]}"; do [[ "${ORDER[$i]}" == "$FROM" ]] && start=$i; done
(( start >= 0 )) || { echo "--from must be one of: ${ORDER[*]}" >&2; exit 2; }
want() { local i; for i in "${!ORDER[@]}"; do [[ "${ORDER[$i]}" == "$1" ]] && (( i >= start )) && return 0; done; return 1; }

A100=(-p gpu --gres=gpu:1 "--constraint=[a100|l40s]" --cpus-per-task=8 --mem=64G)
CHEAP=(-p gpu --gres=gpu:1 "--constraint=[2080_ti|1080_ti|titan_x|m40|v100]" --cpus-per-task=4 --mem=16G)
LOG="$OGAF_DATA/pipeline_$(date -u +%Y%m%dT%H%M%S).jids"; : > "$LOG"
q() {  # q <label> <sbatch args...> -> prints job id; records it
  local label=$1; shift
  if (( DRY )); then echo "DRY sbatch $*" >&2; echo "0"; return; fi
  local id; id=$(sbatch --parsable "$@"); echo "$label=$id" >> "$LOG"; echo "$id"; }
dep() { local d=""; for j in "$@"; do [[ -n "$j" && "$j" != 0 ]] && d="$d:$j"; done; [[ -n "$d" ]] && echo "--dependency=afterok${d}"; }
PREV=""   # job(s) the next stage waits on

if want corpus; then
  C=$(q corpus --array=0-59 unity/corpus.sbatch)
  M=$(q frames $(dep $C) --time=01:00:00 --job-name=ogaf-frames unity/run.sbatch python idm_data.py)
  V=$(q gate   $(dep $C) --time=01:00:00 --job-name=ogaf-gate   unity/run.sbatch python validate_corpus.py)
  E0=$(q exp0  --time=01:30:00 --cpus-per-task=8 --mem=32G --job-name=ogaf-exp0 unity/run.sbatch python exp_0_oracles.py --n 700 --seeds 3 --out "$OGAF_RESULTS/exp_0_repro")
  SUB=$(q subset $(dep $E0) --time=00:20:00 --job-name=ogaf-subset unity/run.sbatch python verification_subset.py --exp0 "$OGAF_RESULTS/exp_0_repro")
  PREV="$M $V"
fi
if want idm; then
  PS=$(q probes-s1 $(dep $PREV) -p gpu --time=02:00:00 --job-name=ogaf-probes-s1 unity/probes.sbatch --smoke)
  PB=$(q probes    $(dep $PS) unity/probes.sbatch)
  AS=$(q idm-s1    $(dep $PREV) -p gpu --time=01:30:00 --job-name=ogaf-idm-s1 unity/train_idm.sbatch --smoke)
  AA=$(q idm       $(dep $AS) --array=0-29 -p gpu,gpu-preempt --time=02:00:00 unity/train_idm.sbatch)
  PREV="$PB $AA"
fi
if want floors; then
  D=$(q floors $(dep $PREV) --time=00:30:00 --job-name=ogaf-expd  unity/run.sbatch python exp_d_floors.py)
  P=$(q power  $(dep $D)    --time=00:30:00 --job-name=ogaf-power unity/run.sbatch python power.py)
  PREV="$P"
  echo "After 'power' finishes: write prereg.md (see prereg_DRAFT.md), commit, push, then:"
  echo "    bash unity/pipeline.sh --from prereg-check"
  [[ "$FROM" == "floors" || "$FROM" == "corpus" || "$FROM" == "idm" ]] && { echo "jids: $LOG"; exit 0; }
fi
if want prereg-check; then
  python prereg_lock.py || { echo "prereg lock FAILED; nothing below is submitted" >&2; exit 1; }
fi
if want e; then
  EJ=$(q expe  $(dep $PREV) "${A100[@]}" --time=04:00:00 --job-name=ogaf-expe unity/run.sbatch python exp_e_confound.py --num-workers 8)
  F1=$(q expf-gen "${CHEAP[@]}" --mem=32G --time=04:00:00 --job-name=ogaf-expf-gen unity/run.sbatch python exp_f_friction.py --generate --clips)
  F2=$(q expf  $(dep $F1) "${A100[@]}" --time=02:00:00 --job-name=ogaf-expf unity/run.sbatch python exp_f_friction.py)
  G=$(q expg   $(dep $EJ) "${A100[@]}" --time=02:00:00 --job-name=ogaf-expg unity/run.sbatch python exp_g_lipschitz.py)
  AN=$(q analyze $(dep $EJ $F2 $G) --time=00:45:00 --job-name=ogaf-analyze unity/run.sbatch python analyze.py --exp all)
  PREV="$AN"
fi
if want h; then
  F3=$(OGAF_CORPUS_ARGS="--friction-mult 3.0 --conditions INTERACT" q fm3corpus --array=0-59 unity/corpus.sbatch)
  CL=""
  for g in box sphere cylinder; do
    for cond in INTERACT ABSENT; do
      CL="$CL $(q wmclip "${CHEAP[@]}" --time=02:00:00 --job-name=ogaf-wmclip unity/run.sbatch python wm/render_clips.py --geometry $g --shards $(seq 0 19) --condition $cond)"
    done
    CL="$CL $(q wmclip3 $(dep $F3) "${CHEAP[@]}" --time=02:00:00 --job-name=ogaf-wmclip3 unity/run.sbatch python wm/render_clips.py --geometry $g --shards $(seq 0 19) --friction-mult 3.0)"
  done
  VAE=$(q vae $(dep $CL) "${A100[@]}" --time=04:00:00 --job-name=ogaf-vae unity/run.sbatch python wm/vae.py)
  SM=$(q wm-s1 $(dep $VAE) -p gpu --time=01:30:00 --job-name=ogaf-wm-s1 unity/wm_train.sbatch --smoke)
  CA=""; for m in WM-base WM-data-poor WM-physics-corrupted; do CA="$CA $(q wm-cache $(dep $SM) "${A100[@]}" --time=01:00:00 --job-name=ogaf-wm-cache unity/run.sbatch python wm/wm_train.py --model $m --cache-only)"; done
  TR=$(q wm-train $(dep $CA) --array=0-2 -p gpu,gpu-preempt unity/wm_train.sbatch)
  PREV="$TR"
fi
if want gen; then
  S1G=$(OGAF_WM_MODEL=WM-base-10 q gen-s1 $(dep $PREV) -p gpu --time=00:30:00 --job-name=ogaf-gen-s1 unity/wm_generate.sbatch --s1)
  VAL=""
  for m in WM-base-100 WM-base-30 WM-base-10 WM-data-poor WM-physics-corrupted; do
    g1=$(OGAF_WM_MODEL=$m q gen-$m $(dep $S1G) --array=0-9 -p gpu,gpu-preempt unity/wm_generate.sbatch)
    g2=$(OGAF_WM_MODEL=$m OGAF_WM_CONDITION=ABSENT OGAF_WM_N=300 q genabs-$m $(dep $S1G) --array=0-4 -p gpu,gpu-preempt unity/wm_generate.sbatch)
    VAL="$VAL $(q val-$m $(dep $g1 $g2) --time=00:30:00 --job-name=ogaf-wm-val unity/run.sbatch python unity/validate.py --dir "$OGAF_DATA/generated/$m/box" --expected-frames 16 --manifest "$OGAF_DATA/generated/$m/box/manifest.json")"
  done
  H=$(q exph $(dep $VAL) "${A100[@]}" --time=03:00:00 --job-name=ogaf-exph unity/run.sbatch python wm/exp_h.py)
  q analyze-h $(dep $H) --time=00:45:00 --job-name=ogaf-analyze unity/run.sbatch python analyze.py --exp all >/dev/null
fi
echo "submitted; job ids in $LOG"
