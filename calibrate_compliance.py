"""Choose the compliant-arm gain by the rule in prereg_addendum_A3.md §K.1.

For each candidate position gain, simulate INTERACT rollouts (no rendering) with
OGAF_ARM_CONTROL=position and T9 recorded but not excluding, and measure how far
contact deflects the arm from the same command executed without contact (the
T9 value, max over the rollout).  Select the STIFFEST gain whose median
deflection is at least one video frame of object motion (pooled delta_pos_min)
in every geometry, with at least 80 % of rollouts passing every other
validator.  Each gain runs in its own process (config reads the gain at import).

    python calibrate_compliance.py          # CPU; writes results/exp_k_calibration
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import config as C

CHILD = r"""
import json, sys, numpy as np
import config as C, scene
thr = C.load_thresholds()
out = {}
for g in C.GEOMETRIES:
    devs, keep = [], []
    for ti in range(int(sys.argv[1])):
        st = scene.tuple_streams(C.MASTER_SEED, g, ti)
        a = scene.sample_action(st["action"])
        r = scene.rollout(a, seed=ti, condition="INTERACT", geometry=g,
                          thresholds=thr.for_geometry(g), master_seed=C.MASTER_SEED, render=False)
        v = r["validators"]
        devs.append(v["T9_arm_deviation"]["value"])
        keep.append(all(x["passed"] for k, x in v.items()
                        if isinstance(x, dict) and k != "T9_arm_deviation"))
    out[g] = {"median_deflection_m": float(np.median(devs)), "p90_deflection_m":
              float(np.percentile(devs, 90)), "keep": float(np.mean(keep)), "n": len(devs)}
print("JSON" + json.dumps(out))
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gains", type=float, nargs="+", default=[40000, 10000, 2500, 625])
    ap.add_argument("--n", type=int, default=60, help="tuples per geometry")
    ap.add_argument("--min-keep", type=float, default=0.80)
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "exp_k_calibration")
    args = ap.parse_args()

    delta = float(C.load_thresholds().pooled_delta_pos_min)
    res = {"rule": "prereg_addendum_A3.md §K.1", "delta_m": delta, "candidates": {}}
    for kp in args.gains:
        env = dict(os.environ, OGAF_ARM_CONTROL="position", OGAF_ARM_KP=str(kp),
                   OGAF_T9_EXCLUDES="0")
        r = subprocess.run([sys.executable, "-c", CHILD, str(args.n)], env=env,
                           capture_output=True, text=True, cwd=C.REPO_ROOT)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("JSON")]
        if r.returncode != 0 or not line:
            print(r.stderr[-2000:]); raise SystemExit(f"kp={kp} failed")
        per = json.loads(line[0][4:])
        ok = all(p["median_deflection_m"] >= delta and p["keep"] >= args.min_keep
                 for p in per.values())
        res["candidates"][f"{kp:g}"] = {"kp": kp, "per_geometry": per, "qualifies": ok}
        print(f"kp={kp:<7g} " + "  ".join(f"{g}: defl {1e3 * p['median_deflection_m']:.2f} mm "
                                          f"keep {p['keep']:.2f}" for g, p in per.items())
              + ("  QUALIFIES" if ok else ""), flush=True)
    q = [c["kp"] for c in res["candidates"].values() if c["qualifies"]]
    res["selected_kp"] = max(q) if q else None
    print(f"\nSELECTED gain: {res['selected_kp'] if q else 'NONE -- Experiment K stops (§K.1)'}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(res, indent=2))
    return 0 if q else 3


if __name__ == "__main__":
    raise SystemExit(main())
