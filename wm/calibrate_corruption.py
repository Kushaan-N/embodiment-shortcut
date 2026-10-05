"""Pick the H2 physics corruption by the rule in prereg_addendum_H2.md §2.

Ground truth only -- no IDM, no metric outcome is computed here:
  D(k)  object-region error between the REAL x k and REAL x1 WM clips of the
        same box tuples (what a perfect wrong-physics world model would score);
  T     1.25 x the object-region error of the existing WM-data-poor generations;
  keep  fraction of x k rollouts passing every validator, per geometry.
Select the qualifying k (D >= T, keep >= 0.80 everywhere) with the smallest
|ln k|, ties to the smaller k.  Writes results/exp_h2_calibration/results.json.

Needs (built by unity/pipeline.sh --from h2-calibrate): corpus shards 0-1 at
every candidate friction (INTERACT, all geometries) and their box WM clips under
wm_clips_fm<k>/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import datasets as ds  # noqa: E402
import provenance  # noqa: E402
from wm.exp_h import load_generated, load_real_clips, object_region_error  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=float, nargs="+", default=[0.1, 0.25, 4.0, 10.0])
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--factor", type=float, default=1.25)
    ap.add_argument("--min-keep", type=float, default=0.80)
    ap.add_argument("--clips", type=Path, default=C.DATA_ROOT / "wm_clips")
    ap.add_argument("--generated-root", type=Path, default=C.DATA_ROOT / "generated")
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "exp_h2_calibration")
    args = ap.parse_args()

    # ---- threshold from the weakest correct-physics model ------------------
    g = load_generated("WM-data-poor", "box", args.generated_root)
    absent = load_real_clips(g["tuple_index"], "box", "ABSENT", args.clips)
    e_dp = float(np.nanmean(object_region_error(g["generated"], g["real"], absent)))
    T = args.factor * e_dp
    print(f"WM-data-poor object-region error = {e_dp:.5f}  ->  threshold T = {T:.5f}")

    lo, hi = min(args.shards) * C.SHARD_SIZE, (max(args.shards) + 1) * C.SHARD_SIZE
    cands = {}
    for k in args.candidates:
        fm = f"_fm{k:g}"
        root_k = args.clips.parent / f"{args.clips.name}{fm}"
        ids = [t for t in range(lo, hi)
               if (root_k / "box" / "INTERACT" / f"{t:07d}.npz").exists()
               and (args.clips / "box" / "INTERACT" / f"{t:07d}.npz").exists()]
        if not ids:
            raise SystemExit(f"x{k:g}: no box clips under {root_k}; build them first")
        xk = load_real_clips(ids, "box", "INTERACT", root_k)
        x1 = load_real_clips(ids, "box", "INTERACT", args.clips)
        ab = load_real_clips(ids, "box", "ABSENT", args.clips)
        D = float(np.nanmean(object_region_error(xk, x1, ab)))
        keep = {}
        for geo in C.GEOMETRIES:
            v = []
            for s in args.shards:
                p = ds.shard_path(C.CORPUS_ROOT, geo, "INTERACT", s, k)
                if not p.exists():
                    raise SystemExit(f"{p} missing; build the x{k:g} shards first")
                v.append(np.load(p)["validators_passed"])
            keep[geo] = float(np.concatenate(v).mean())
        ok = D >= T and min(keep.values()) >= args.min_keep
        cands[f"{k:g}"] = {"k": k, "n_tuples": len(ids), "D": D, "D_over_T": D / T,
                           "keep": keep, "qualifies": bool(ok)}
        print(f"  x{k:<5g} D={D:.5f} ({D / T:.2f} T)  keep={keep}  {'QUALIFIES' if ok else '-'}")

    q = [c for c in cands.values() if c["qualifies"]]
    sel = min(q, key=lambda c: (abs(math.log(c["k"])), c["k"]))["k"] if q else None
    res = {"rule": "prereg_addendum_H2.md §2", "data_poor_object_error": e_dp,
           "threshold_T": T, "factor": args.factor, "min_keep": args.min_keep,
           "candidates": cands, "selected_k": sel}
    print(f"\nSELECTED friction multiplier: {sel if sel is not None else 'NONE -- H2 stops (§2)'}")
    args.out.mkdir(parents=True, exist_ok=True)
    provenance.save_run(args.out, "exp_h2_calibration", {"dummy": np.zeros(1)},
                        extra={"results": res}, arrays_name="empty.npz")
    (args.out / "results.json").write_text(json.dumps(res, indent=2))
    if sel is not None:
        write_ladder_v2(sel)
    return 0 if sel is not None else 3


def write_ladder_v2(k: float) -> Path:
    """wm/ladder_v2.yaml = wm/ladder.yaml with model 5 replaced by the selected
    corruption, the object-region gate, and the addendum that seals it."""
    import yaml

    src = Path(__file__).resolve().parent / "ladder.yaml"
    doc = yaml.safe_load(src.read_text())
    doc["models"] = [m for m in doc["models"] if m["name"] != "WM-physics-corrupted"] + [{
        "id": 5, "name": "WM-physics-corrupted-v2", "from": "base",
        "overrides": {"train": {"friction_mult": float(k)}},
        "note": f"friction x{k:g}, selected by wm/calibrate_corruption.py under "
                f"prereg_addendum_H2.md §2",
    }]
    doc["ladder_gt"] = "object_region"
    doc["c3b"] = {"base": "WM-base-100", "corrupted": "WM-physics-corrupted-v2"}
    doc["addendum"] = "prereg_addendum_H2.md"
    out = src.with_name("ladder_v2.yaml")
    out.write_text("# GENERATED by wm/calibrate_corruption.py -- do not edit by hand.\n"
                   "# prereg_addendum_H2.md; selected friction x" + f"{k:g}\n"
                   + yaml.safe_dump(doc, sort_keys=False))
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
