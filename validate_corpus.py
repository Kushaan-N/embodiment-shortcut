"""Re-run every §6 validator, the T11 gate, and split integrity over a finished corpus.

Runs offline against a corpus directory -- no re-simulation.  Everything it
checks was recorded per rollout at generation time, which is the point of
putting measured values (not just booleans) into ``validators`` (§0.4).

Three families of check:

1. **Per-rollout validators** (T5, T9, T10, in-frustum, settling).  Aggregated
   into pass rates and exclusion counts per (geometry, condition), because the
   pre-registered inclusion policy is "excluded and counted", not "excluded".

2. **T11 placement-independence gate.**  Ridge regression from DECOY position
   to each action dimension; ASSERT R^2 < 0.01 per dimension.  Run *before any
   IDM trains*: discovering a placement bug after 50 trainings wastes
   everything (§4-T11).

3. **T7 split integrity.**  No ``(geometry, tuple_index)`` key may appear in
   two splits.

    python validate_corpus.py [--root data/corpus]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import config as C
import datasets as ds
import provenance
import scene

EXPERIMENT = "validate_corpus"


# ==========================================================================
# 1. Per-rollout validators
# ==========================================================================


def aggregate_validators(root: Path) -> dict:
    per_cell: dict = defaultdict(lambda: defaultdict(
        lambda: {"n": 0, "failed": 0, "values": []}))
    totals: dict = defaultdict(lambda: {"n": 0, "kept": 0})

    for path, _ in ds.iter_records(root):
        vpath = path.with_suffix(".validators.json")
        if not vpath.exists():
            continue
        rows = json.loads(vpath.read_text())
        for row in rows:
            cell = (row["geometry"], row["condition"])
            totals[cell]["n"] += 1
            all_ok = True
            for name, v in row["validators"].items():
                slot = per_cell[cell][name]
                slot["n"] += 1
                if not v.get("passed", True):
                    slot["failed"] += 1
                    all_ok = False
                if v.get("value") is not None:
                    slot["values"].append(float(v["value"]))
            totals[cell]["kept"] += int(all_ok)

    out = {}
    for cell, checks in per_cell.items():
        key = f"{cell[0]}/{cell[1]}"
        out[key] = {"n_rollouts": totals[cell]["n"], "n_kept": totals[cell]["kept"],
                    "exclusion_rate": 1.0 - totals[cell]["kept"] / max(totals[cell]["n"], 1),
                    "checks": {}}
        for name, slot in checks.items():
            vals = np.asarray(slot["values"]) if slot["values"] else np.array([np.nan])
            out[key]["checks"][name] = {
                "n": slot["n"], "n_failed": slot["failed"],
                "fail_rate": slot["failed"] / max(slot["n"], 1),
                "value_median": float(np.nanmedian(vals)),
                "value_p95": float(np.nanpercentile(vals, 95)),
                "value_max": float(np.nanmax(vals)),
            }
    return out


# ==========================================================================
# 2. T11 placement-independence gate
# ==========================================================================


def t11_gate(root: Path, alpha: float = 1.0) -> dict:
    """ASSERT: DECOY position predicts no action dimension (R^2 < 0.01).

    Ridge regression on the generated corpus, per §4-T11.  Reports R^2 per
    dimension per geometry; the gate is the maximum over all of them.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict

    per_geom = {}
    worst = 0.0
    for geometry in C.GEOMETRIES:
        xs, ys = [], []
        for _, z in ds.iter_records(root, geometry=geometry, condition="DECOY"):
            if "decoy_xy" not in z.files:
                continue
            xs.append(z["decoy_xy"])
            ys.append(z["action"])
        if not xs:
            per_geom[geometry] = {"skipped": "no DECOY rollouts found"}
            continue
        X = np.concatenate(xs).astype(np.float64)
        Y = np.concatenate(ys).astype(np.float64)
        # Include radius and angle as well as (x, y): a placement bug could be
        # radial or angular and invisible to a purely linear fit on (x, y).
        feats = np.column_stack([
            X, np.linalg.norm(X, axis=1), np.arctan2(X[:, 1], X[:, 0]),
            np.sin(np.arctan2(X[:, 1], X[:, 0])), np.cos(np.arctan2(X[:, 1], X[:, 0])),
        ])
        dims = {}
        for j, name in enumerate(C.ACTION_DIMS):
            y = Y[:, j]
            pred = cross_val_predict(Ridge(alpha=alpha), feats, y, cv=5)
            ss_res = float(((y - pred) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
            dims[name] = {"r2": r2, "n": int(len(y))}
            worst = max(worst, r2)
        per_geom[geometry] = dims

    tol = C.TOLERANCES.decoy_placement_r2_max
    return {
        "name": "T11_decoy_placement_independent_of_action",
        "passed": bool(worst < tol),
        "value": float(worst),
        "threshold": float(tol),
        "per_geometry": per_geom,
        "note": "cross-validated ridge R^2 from DECOY placement to each action dim; "
                "run BEFORE any IDM trains (§4-T11)",
    }


# ==========================================================================
# 3. T7 split integrity
# ==========================================================================


def split_integrity(root: Path) -> dict:
    records = []
    for _, z in ds.iter_records(root):
        for ti, sp, g in zip(z["tuple_index"], z["split"].astype(str),
                             z["geometry"].astype(str)):
            records.append({"geometry": g, "tuple_index": int(ti), "split": sp})
    out = ds.assert_no_tuple_spans_splits(records)
    counts: dict = defaultdict(lambda: defaultdict(int))
    for r in records:
        counts[r["geometry"]][r["split"]] += 1
    out["counts_by_geometry"] = {g: dict(v) for g, v in counts.items()}
    return out


# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=C.CORPUS_ROOT)
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "validate_corpus")
    args = ap.parse_args()

    print(f"validating corpus at {args.root}")
    validators = aggregate_validators(args.root)
    t11 = t11_gate(args.root)
    splits = split_integrity(args.root)
    geometry_checks = [scene.assert_corridor_decoy_disjoint(g) for g in C.GEOMETRIES]

    print("\n--- per-rollout validators (exclusions are counted, not hidden) ---")
    for cell, v in sorted(validators.items()):
        print(f"{cell:20s} n={v['n_rollouts']:6d} kept={v['n_kept']:6d} "
              f"excluded={v['exclusion_rate'] * 100:5.2f}%")
        for name, c in sorted(v["checks"].items()):
            if c["n_failed"]:
                print(f"    {name:32s} fail={c['n_failed']:5d} "
                      f"({c['fail_rate'] * 100:5.2f}%)  median={c['value_median']:.4g} "
                      f"max={c['value_max']:.4g}")

    print("\n--- T11 placement-independence gate ---")
    print(f"  max cross-validated R^2 = {t11['value']:.5f}  (threshold {t11['threshold']})  "
          f"{'PASS' if t11['passed'] else 'FAIL'}")
    for g, dims in t11["per_geometry"].items():
        if "skipped" in dims:
            print(f"    {g}: {dims['skipped']}")
            continue
        print("    " + g + ": " + ", ".join(f"{k}={v['r2']:+.5f}" for k, v in dims.items()))

    print("\n--- T7 split integrity ---")
    print(f"  {'PASS' if splits['passed'] else 'FAIL'}  "
          f"tuples checked={splits['n_tuples_checked']}  offenders={splits['value']:.0f}")
    for g, cnt in splits["counts_by_geometry"].items():
        print(f"    {g}: {cnt}")

    print("\n--- scene-level geometry checks ---")
    for gc in geometry_checks:
        print(f"  {gc['geometry']:9s} corridor/DECOY clearance="
              f"{gc['clearance_m'] * 1e3:.1f} mm  {'PASS' if gc['passed'] else 'FAIL'}")

    all_validators_ok = all(
        c["n_failed"] == 0 or name.startswith("T5_occlusion")
        for v in validators.values() for name, c in v["checks"].items()
    )
    passed = bool(t11["passed"] and splits["passed"]
                  and all(g["passed"] for g in geometry_checks))

    report = {
        "passed": passed,
        "all_per_rollout_validators_clean": all_validators_ok,
        "validators": validators,
        "t11_gate": t11,
        "split_integrity": splits,
        "geometry_checks": geometry_checks,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "report.json", "w") as fh:
        json.dump(provenance._jsonable(report), fh, indent=2, sort_keys=True)
    provenance.save_run(args.out, EXPERIMENT, {"dummy": np.zeros(1)},
                        extra={"report": report}, arrays_name="empty.npz")

    print("\n" + "=" * 72)
    if passed:
        print("CORPUS GATE: PASS -- proceed to IDM training (§16 step 8).")
    else:
        print("CORPUS GATE: FAIL -- STOP (§0.1).  Do not train on this corpus.\n"
              "  A T11 failure means DECOY placement leaks the action: fix the sampler\n"
              "  and regenerate.  A T7 failure means paired rollouts span splits, which\n"
              "  biases G TOWARD the hypothesis.")
    print("=" * 72)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
