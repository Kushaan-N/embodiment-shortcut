"""Resolution curve: how large a physics change must be before each metric sees it.

POST-HOC, EXPLORATORY (not in prereg.md or prereg_addendum_H2.md).  Uses only
corpora and checkpoints that already exist -- no new simulation, no training.

For every held-out INTERACT tuple and every friction multiplier k on disk, the
change of physics moves the object's SETTLED position by a known amount
(``||p_k(s_del) - p_1(s_del)||``, from the simulator).  The same trained IDMs
(5 seeds each, error averaged over seeds) score the tuple at x1 and at xk; the
per-tuple error change is what the metric "sees" of that physics change.
Pooling (tuple, k) pairs and binning by displacement gives, per metric, the
error change as a function of how far the object went -- and the smallest
displacement it reliably detects (bootstrap 95 % CI of the mean change > 0).

    python resolution_curve.py          # GPU (scores clips); writes results/exp_resolution
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config as C
import datasets as ds
import exp_d_floors as expd
import exp_f_friction as expf
import idm_data as idd
import provenance

EXPERIMENT = "exp_resolution"
BIN_EDGES_MM = (0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, np.inf)


def settled_positions(geometry: str, fm: float) -> dict:
    """(geometry, tuple_index) -> object position (m) at that rollout's own s_del."""
    out = {}
    for _, z in ds.iter_records(C.CORPUS_ROOT, geometry=geometry, condition="INTERACT",
                                friction_mult=fm):
        p = z["obj_poses_horizons"][:, ds.HORIZONS.index("s_del"), :3]
        for ti, ok, pos in zip(z["tuple_index"], z["validators_passed"], p):
            if ok:
                out[(geometry, int(ti))] = np.asarray(pos, dtype=np.float64)
    return out


def seed_mean_scores(seed_dirs: dict, geometries, fm: float) -> dict:
    acc: dict = {}
    for _, d in sorted(seed_dirs.items()):
        ev = expf._evaluate_at_friction(d, geometries, fm)
        for g, ti, e in zip(ev["geometry"], ev["tuple_index"], ev["per_sample_mae"]):
            acc.setdefault((str(g), int(ti)), []).append(float(e))
    n = len(seed_dirs)
    return {k: float(np.mean(v)) for k, v in acc.items() if len(v) == n}


def boot_mean_ci(x: np.ndarray, rng, n_boot: int) -> list:
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--multipliers", type=float, nargs="+", default=[0.1, 0.25, 0.5, 2.0, 3.0, 4.0])
    ap.add_argument("--variants", nargs="+", default=["A-std", "A-del"])
    ap.add_argument("--geometries", nargs="+", default=list(C.GEOMETRIES))
    ap.add_argument("--min-bin", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / EXPERIMENT)
    args = ap.parse_args()

    for fm in args.multipliers:                    # idempotent; rebuilt if stale
        for g in args.geometries:
            idd.materialise_frames(g, "INTERACT", friction_mult=fm)

    runs = expd.discover_runs(C.CHECKPOINT_ROOT, args.variants)
    pos = {fm: {} for fm in [1.0] + args.multipliers}
    for fm in pos:
        for g in args.geometries:
            pos[fm].update(settled_positions(g, fm))
    scores = {v: {fm: seed_mean_scores(runs[v], args.geometries, fm) for fm in pos}
              for v in args.variants}
    for v in args.variants:
        print(f"{v}: {len(runs[v])} seeds; tuples scored per k: "
              f"{ {f'{fm:g}': len(s) for fm, s in scores[v].items()} }", flush=True)

    rows = []                                      # one per (geometry, tuple, k)
    for fm in args.multipliers:
        keys = set(pos[fm]) & set(pos[1.0])
        for v in args.variants:
            keys &= set(scores[v][fm]) & set(scores[v][1.0])
        for key in sorted(keys):
            rows.append({"geometry": key[0], "tuple_index": key[1], "k": fm,
                         "disp_mm": 1e3 * float(np.linalg.norm(pos[fm][key] - pos[1.0][key])),
                         **{f"d_{v}": scores[v][fm][key] - scores[v][1.0][key]
                            for v in args.variants}})
    disp = np.array([r["disp_mm"] for r in rows])
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    prior = 0.25
    bins = []
    for lo, hi in zip(BIN_EDGES_MM[:-1], BIN_EDGES_MM[1:]):
        sel = (disp >= lo) & (disp < hi)
        n = int(sel.sum())
        b = {"lo_mm": lo, "hi_mm": None if np.isinf(hi) else hi, "n": n,
             "mean_disp_mm": float(disp[sel].mean()) if n else None}
        for v in args.variants:
            x = np.array([r[f"d_{v}"] for r, s in zip(rows, sel) if s])
            if n >= args.min_bin:
                sd = x.std(ddof=1)
                b[v] = {"mean_delta": float(x.mean()), "delta_pct_prior": 100 * float(x.mean()) / prior,
                        "ci": boot_mean_ci(x, rng, args.n_boot),
                        "d": float(x.mean() / sd) if sd > 0 else 0.0}
        bins.append(b)

    thresholds = {}
    for v in args.variants:
        ok = [b for b in bins if v in b]
        thr = None
        for i, b in enumerate(ok):                 # smallest bin from which ALL larger bins detect
            if all(bb[v]["ci"][0] > 0 for bb in ok[i:]):
                thr = b["lo_mm"]
                break
        thresholds[v] = thr

    print(f"\n{len(rows)} (tuple, k) pairs.  delta_pos_min (one video frame) = "
          f"{1e3 * float(np.mean(list(C.load_thresholds().delta_pos_min.values()))):.1f} mm\n")
    hdr = f"{'displacement':>16s} {'n':>5s}" + "".join(f" {v + ' Δ%prior [CI]':>34s}" for v in args.variants)
    print(hdr)
    for b in bins:
        lab = f"{b['lo_mm']:g}-{b['hi_mm']:g} mm" if b["hi_mm"] else f">{b['lo_mm']:g} mm"
        cells = ""
        for v in args.variants:
            if v in b:
                c = b[v]
                cells += (f" {c['delta_pct_prior']:+8.2f} [{100 * c['ci'][0] / prior:+6.2f},"
                          f"{100 * c['ci'][1] / prior:+6.2f}] d={c['d']:+.2f}")
            else:
                cells += f" {'(n < ' + str(args.min_bin) + ')':>34s}"
        print(f"{lab:>16s} {b['n']:5d}{cells}")
    for v, t in thresholds.items():
        print(f"  detection threshold {v}: {'>= ' + format(t, 'g') + ' mm' if t is not None else 'none in range'}")

    res = {"note": "POST-HOC, EXPLORATORY: not pre-registered", "multipliers": args.multipliers,
           "n_pairs": len(rows), "bins": bins, "detection_threshold_mm": thresholds,
           "bin_edges_mm": [e if np.isfinite(e) else None for e in BIN_EDGES_MM]}
    arrays = {"disp_mm": disp, "k": np.array([r["k"] for r in rows]),
              "tuple_index": np.array([r["tuple_index"] for r in rows]),
              "geometry": np.array([r["geometry"] for r in rows])}
    for v in args.variants:
        arrays[f"delta_{v}"] = np.array([r[f"d_{v}"] for r in rows])
    args.out.mkdir(parents=True, exist_ok=True)
    provenance.save_run(args.out, EXPERIMENT, arrays, extra={"results": res}, arrays_name="resolution.npz")
    (args.out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
