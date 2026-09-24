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
# 4. Completeness -- the gate must describe the corpus it sits next to
# ==========================================================================


def corpus_completeness(root: Path, expect: int, friction_mult: float = 1.0) -> dict:
    """Every (geometry, condition) cell present with tuple indices exactly
    ``0..expect-1`` (no gaps, duplicates or stray shards), and every shard
    carrying its ``.validators.json``.

    Without this the gate PASSED on a 24-tuple box-only smoke corpus with
    two geometries "skipped" (results/validate_corpus/report.json, 2026-08-17)
    -- a report that certified nothing about the data it sat next to.
    """
    cells: dict = {}
    missing_json = []
    for path, z in ds.iter_records(root, friction_mult=friction_mult):
        key = f"{str(z['geometry'][0])}/{str(z['condition'][0])}"
        cells.setdefault(key, []).extend(int(t) for t in z["tuple_index"])
        if not path.with_suffix(".validators.json").exists():
            missing_json.append(str(path))
    expected_cells = [f"{g}/{c}" for g in C.GEOMETRIES for c in C.CONDITIONS]
    missing_cells = [k for k in expected_cells if k not in cells]
    wrong = {}
    for k, tis in cells.items():
        s = sorted(tis)
        if s != list(range(expect)):
            wrong[k] = {"n": len(s), "n_unique": len(set(s)),
                        "min": s[0] if s else None, "max": s[-1] if s else None}
    return {"name": "corpus_completeness", "expected_tuples_per_cell": int(expect),
            "cells_present": sorted(cells), "missing_cells": missing_cells,
            "cells_with_wrong_tuples": wrong,
            "shards_missing_validators_json": missing_json,
            "passed": bool(not missing_cells and not wrong and not missing_json)}


def clip_completeness(root: Path, clip_root: Path, friction_mult: float = 1.0) -> dict:
    """Every clip variant must exist for every tuple: the first consumer to
    notice a missing clip is otherwise ``ClipDataset.__getitem__`` on a GPU."""
    checked = missing = 0
    examples: list = []
    for _, z in ds.iter_records(root, friction_mult=friction_mult):
        g, c = str(z["geometry"][0]), str(z["condition"][0])
        for ti in z["tuple_index"]:
            for v in C.CLIP_VARIANTS:
                checked += 1
                p = ds.clip_path(clip_root, v, g, c, int(ti), friction_mult)
                if not p.exists():
                    missing += 1
                    if len(examples) < 5:
                        examples.append(str(p))
    return {"name": "clip_completeness", "clip_root": str(clip_root),
            "variants": list(C.CLIP_VARIANTS), "checked": checked, "missing": missing,
            "examples": examples, "passed": bool(checked > 0 and missing == 0)}


# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=C.CORPUS_ROOT)
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "validate_corpus")
    ap.add_argument("--expect-tuples", type=int, default=C.N_TUPLES_PER_GEOMETRY,
                    help="tuples per (geometry, condition) cell the gate certifies "
                         "(default: the pre-registered corpus size)")
    ap.add_argument("--clips", type=Path, default=C.CLIP_ROOT,
                    help="clip store to check for completeness (A-std + A-clip-del)")
    ap.add_argument("--no-clips", action="store_true",
                    help="skip the clip check (a corpus that will NOT train Architecture A)")
    ap.add_argument("--exclusion-cap", type=float, default=0.05,
                    help="max per-validator fail rate before the gate refuses "
                         "(unity/README.md: >5%% failures -> diagnose before analysing)")
    args = ap.parse_args()

    print(f"validating corpus at {args.root}")
    validators = aggregate_validators(args.root)
    t11 = t11_gate(args.root)
    splits = split_integrity(args.root)
    geometry_checks = [scene.assert_corridor_decoy_disjoint(g) for g in C.GEOMETRIES]
    completeness = corpus_completeness(args.root, args.expect_tuples)
    clips = None if args.no_clips else clip_completeness(args.root, args.clips)

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

    print("\n--- completeness ---")
    print(f"  cells present: {len(completeness['cells_present'])}/"
          f"{len(C.GEOMETRIES) * len(C.CONDITIONS)}  expected tuples/cell="
          f"{completeness['expected_tuples_per_cell']}  "
          f"{'PASS' if completeness['passed'] else 'FAIL'}")
    for k in completeness["missing_cells"]:
        print(f"    MISSING cell {k}")
    for k, v in completeness["cells_with_wrong_tuples"].items():
        print(f"    {k}: {v}")
    for p in completeness["shards_missing_validators_json"][:5]:
        print(f"    shard without validators.json: {p}")
    if clips is not None:
        print(f"  clips: {clips['missing']} missing of {clips['checked']} checked under "
              f"{clips['clip_root']}  {'PASS' if clips['passed'] else 'FAIL'}")
        for p in clips["examples"]:
            print(f"    missing {p}")

    all_validators_ok = all(
        c["n_failed"] == 0 or name.startswith("T5_occlusion")
        for v in validators.values() for name, c in v["checks"].items()
    )
    # Exclusions are counted, not hidden -- but a validator failing on more
    # than the cap is a bug to diagnose BEFORE training, not a footnote.
    over_cap = {f"{cell}:{name}": c["fail_rate"]
                for cell, v in validators.items() for name, c in v["checks"].items()
                if not name.startswith("T5_occlusion") and c["fail_rate"] > args.exclusion_cap}
    for k, r in sorted(over_cap.items()):
        print(f"  EXCLUSION CAP EXCEEDED  {k}: {r * 100:.2f}% > {args.exclusion_cap * 100:.0f}%")
    t11_skipped = [g for g, d in t11["per_geometry"].items() if "skipped" in d]

    passed = bool(t11["passed"] and not t11_skipped and splits["passed"]
                  and all(g["passed"] for g in geometry_checks)
                  and completeness["passed"]
                  and (clips is None or clips["passed"])
                  and not over_cap)

    report = {
        "passed": passed,
        "all_per_rollout_validators_clean": all_validators_ok,
        "validators": validators,
        "validators_over_exclusion_cap": over_cap,
        "exclusion_cap": args.exclusion_cap,
        "t11_gate": t11,
        "t11_geometries_skipped": t11_skipped,
        "split_integrity": splits,
        "geometry_checks": geometry_checks,
        "completeness": completeness,
        "clips": clips,
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
              "  biases G TOWARD the hypothesis.  A completeness/clip failure means the\n"
              "  corpus on disk is not the corpus the protocol describes: finish the\n"
              "  build (datasets.py ... --clips) before any IDM trains.")
    print("=" * 72)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
