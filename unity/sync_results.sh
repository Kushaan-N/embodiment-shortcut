#!/bin/bash
# Copy the small, versionable result records from the workspace back into the
# repo (results/<exp>/{results,report,floors,power,thresholds}.json + each
# metadata.json) and report workspace expiry.  The workspace has NO backups
# and expires; the JSON records are the evidence the paper rests on.
#
#   source $WS/activate.sh && bash unity/sync_results.sh [--commit]
set -euo pipefail
: "${OGAF_RESULTS:?source unity/env.sh (or the workspace activate.sh) first}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Directories whose COMMITTED record is the original (pre-Unity) measurement;
# the Unity re-runs of these live in results/<name>_repro and must not
# overwrite the originals.
SKIP=" exp_0 verification_subset exp_b exp_c exp_a "
n=0
for d in "$OGAF_RESULTS"/*/; do
  name=$(basename "$d")
  [[ "$SKIP" == *" $name "* ]] && continue
  for f in results.json report.json floors.json power.json thresholds.json metadata.json; do
    [[ -f "$d/$f" ]] || continue
    mkdir -p "$REPO/results/$name"
    if ! cmp -s "$d/$f" "$REPO/results/$name/$f"; then
      # no -p: /work does not support preserving mode and set -e would abort the loop
      cp "$d/$f" "$REPO/results/$name/$f"; echo "updated results/$name/$f"; n=$((n+1))
    fi
  done
done
echo "$n file(s) updated"
if command -v ws_list >/dev/null; then
  ws_list -v 2>/dev/null | awk '/^id: ogaf/{f=1} f&&/remaining time/{print "workspace ogaf:", $0; exit}'
fi
if [[ "${1:-}" == "--commit" && $n -gt 0 ]]; then
  cd "$REPO" && git add -f results/*/*.json && git commit -m "results: sync from workspace ($(date -u +%F))" && echo "committed; push when ready"
fi
