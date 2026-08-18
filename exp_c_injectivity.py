"""Experiment C -- injectivity curve (§9-C).  CPU, minutes.  Gate C.  Before D.

T3: at the delayed horizon, distinct actions can settle the object into
identical states.  Where that happens the action is genuinely unrecoverable and
OG-AF would punish a *correct* world model.  This experiment measures how much
of the action space is affected, as a curve rather than at a single point.

Method
------
Sample action pairs ``(a, a')`` that share a seed -- so the initial object pose
is identical and the only difference is the action -- across a wide range of
separations.  For each pair, measure the ground-truth settled-state separation
using the per-geometry symmetry-quotiented distances of §7.2.

A pair is **unidentifiable at eps_a** if the actions differ by at least
``eps_a`` (in units of each dimension's sampling range) yet the settled states
differ by less than the Experiment A resolution scale in *both* translation and
rotation -- i.e. two meaningfully different actions produced states no
measurement here could tell apart.

The reported quantity is ``P(states indistinguishable | action sep >= eps_a)``
as a function of eps_a, per geometry, at s_std and s_del, and for the box under
both quotiented and unquotiented rotation (§7.2 leaves it to this experiment to
adjudicate which is binding).

    python exp_c_injectivity.py [--pairs 200] [--eps-a 0.10]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config as C
import distances as D
import provenance
import scene

EXPERIMENT = "exp_c_injectivity"

#: Pre-declared before looking at any number (§9-C: the gate may be moved NOW,
#: but is frozen at pre-registration).
GATE_C_MAX_UNIDENTIFIABLE = 0.25
EPS_A_GRID = np.concatenate([np.linspace(0.01, 0.10, 10), np.linspace(0.12, 1.0, 23)])


def sample_pairs(n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """(n_pairs, 2, 3) action pairs spanning separations from ~0 to the full box.

    Separations are log-spaced so the small-eps_a region -- where
    identifiability actually fails -- is well populated rather than swamped by
    uniformly-random far-apart pairs.
    """
    lo, hi = C.action_ranges_array()
    span = hi - lo
    out = np.empty((n_pairs, 2, 3))
    for i in range(n_pairs):
        a = rng.uniform(lo, hi)
        r = 10 ** rng.uniform(np.log10(0.003), np.log10(1.2))
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        b = np.clip(a + r * span * d, lo, hi)
        out[i, 0], out[i, 1] = a, b
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--eps-a", type=float, default=0.10,
                    help="pre-declared working action separation, as a fraction of range")
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "exp_c")
    ap.add_argument("--seed", type=int, default=C.MASTER_SEED + 7)
    args = ap.parse_args()

    thresholds = C.load_thresholds()
    lo, hi = C.action_ranges_array()
    span = hi - lo
    rng = np.random.default_rng(args.seed)
    pairs = sample_pairs(args.pairs, rng)

    arrays: dict = {"pairs": pairs, "eps_a_grid": EPS_A_GRID}
    results: dict = {"eps_a_working": args.eps_a, "eps_a_grid": EPS_A_GRID.tolist(),
                     "gate_threshold": GATE_C_MAX_UNIDENTIFIABLE, "per_geometry": {}}

    for geometry in C.GEOMETRIES:
        thr = thresholds.for_geometry(geometry)
        size = C.GEOM_SIZE[geometry]
        d_pos_min = thr["delta_pos_min"]
        d_rot_min = thr["delta_rot_min"]
        rot_defined = d_rot_min is not None

        rec = {k: [] for k in ("action_sep", "dpos_std", "drot_std_q", "drot_std_u",
                               "dpos_del", "drot_del_q", "drot_del_u", "valid")}
        print(f"  {geometry}: simulating {args.pairs} pairs "
              f"(delta_pos_min={d_pos_min * 1e3:.3f} mm, "
              f"delta_rot_min={'n/a' if not rot_defined else f'{np.degrees(d_rot_min):.3f} deg'})")

        for i, (a, b) in enumerate(pairs):
            # Same seed for both members: identical initial object pose, so the
            # measured state separation is caused by the action alone.
            rs = [scene.rollout(x, seed=i, condition="INTERACT", geometry=geometry,
                                render=False, thresholds=thr) for x in (a, b)]
            ok = all(r["validators"]["_all_passed"] for r in rs)
            rec["valid"].append(ok)
            rec["action_sep"].append(float(np.max(np.abs(a - b) / span)))
            for hname, key in (("s_std", "std"), ("s_del", "del")):
                idx = [r["horizon_idx"][hname] for r in rs]
                p1 = rs[0]["obj_poses"][idx[0]]
                p2 = rs[1]["obj_poses"][idx[1]]
                dp, dr_q = D.pose_distance(p1, p2, geometry, size, quotient=True)
                _, dr_u = D.pose_distance(p1, p2, geometry, size, quotient=False)
                rec[f"dpos_{key}"].append(dp)
                rec[f"drot_{key}_q"].append(dr_q)
                rec[f"drot_{key}_u"].append(dr_u)
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{args.pairs}")

        rec = {k: np.asarray(v) for k, v in rec.items()}
        for k, v in rec.items():
            arrays[f"{geometry}_{k}"] = v
        keep = rec["valid"]

        geom_res = {"n_pairs": int(keep.sum()), "delta_pos_min": d_pos_min,
                    "delta_rot_min": d_rot_min, "curves": {}}

        for horizon in ("std", "del"):
            for quot, qkey in ((True, "q"), (False, "u")):
                dpos = rec[f"dpos_{horizon}"][keep]
                drot = rec[f"drot_{horizon}_{qkey}"][keep]
                asep = rec["action_sep"][keep]
                # "indistinguishable" = below the measurement scale in BOTH
                # channels.  When the quotiented rotation is undefined (sphere)
                # translation alone decides, which is the correct reading of
                # "rotation excluded entirely".
                below_pos = dpos < d_pos_min
                if quot and not rot_defined:
                    # "sphere -> rotation excluded entirely" (§7.2): translation
                    # alone decides indistinguishability.
                    indist = below_pos
                else:
                    ref = d_rot_min if quot else thresholds.delta_rot_raw_min[geometry]
                    indist = below_pos & (drot < ref)

                curve, counts = [], []
                for eps in EPS_A_GRID:
                    sel = asep >= eps
                    counts.append(int(sel.sum()))
                    curve.append(float(indist[sel].mean()) if sel.any() else float("nan"))
                label = f"{horizon}_{'quotiented' if quot else 'unquotiented'}"
                at_working = float(np.interp(args.eps_a, EPS_A_GRID, np.asarray(curve)))
                geom_res["curves"][label] = {
                    "unidentifiable_fraction": curve,
                    "n_pairs_at_eps": counts,
                    "at_working_eps_a": at_working,
                }
                print(f"    {geometry:9s} {label:22s} unidentifiable@eps_a={args.eps_a:.2f}: "
                      f"{at_working * 100:5.1f}%")

        results["per_geometry"][geometry] = geom_res

    gate = _evaluate_gate_c(results, args.eps_a)
    results["_gate_C"] = gate

    args.out.mkdir(parents=True, exist_ok=True)
    provenance.save_run(args.out, EXPERIMENT, arrays, seeds={"pair_seed": args.seed},
                        extra={"eps_a_working": args.eps_a,
                               "gate_threshold": GATE_C_MAX_UNIDENTIFIABLE},
                        arrays_name="injectivity.npz")
    with open(args.out / "results.json", "w") as fh:
        json.dump(provenance._jsonable(results), fh, indent=2, sort_keys=True)

    print("\n" + "=" * 72)
    print(gate["report"])
    print("=" * 72)
    return 0 if gate["passed"] else 2


def _evaluate_gate_c(results: dict, eps_a: float) -> dict:
    lines, ok = [], True
    per_geom = {}
    for g, res in results["per_geometry"].items():
        # The primary curve is the delayed horizon with quotiented rotation:
        # that is the horizon OG-AF uses and the distance §7.2 prescribes.
        frac = res["curves"]["del_quotiented"]["at_working_eps_a"]
        frac_u = res["curves"]["del_unquotiented"]["at_working_eps_a"]
        frac_std = res["curves"]["std_quotiented"]["at_working_eps_a"]
        passed = frac <= GATE_C_MAX_UNIDENTIFIABLE
        ok = ok and passed
        per_geom[g] = {"s_del_quotiented": frac, "s_del_unquotiented": frac_u,
                       "s_std_quotiented": frac_std, "passed": bool(passed),
                       "binding_distance": "quotiented" if frac >= frac_u else "unquotiented"}
        lines.append(f"{g:9s} s_del unidentifiable={frac * 100:5.1f}% "
                     f"(unquotiented {frac_u * 100:5.1f}%, s_std {frac_std * 100:5.1f}%)  "
                     f"{'PASS' if passed else 'FAIL'}")

    report = [f"GATE C: {'PASS' if ok else 'FAIL'}  "
              f"(eps_a={eps_a:.3f}, threshold={GATE_C_MAX_UNIDENTIFIABLE:.2f})", *lines, ""]
    if ok:
        report.append(
            "  Enough of the action space stays identifiable at s_del for OG-AF to have\n"
            "  signal.  Freeze eps_a and these horizon definitions at pre-registration."
        )
    else:
        report.append(
            "  STOP (§0.1).  A large share of action space is unidentifiable at s_del.\n"
            "  Permitted remedies, NOW and not later: shorten the delayed horizon, or\n"
            "  drop the offending geometry.  Do NOT raise the gate threshold, and do not\n"
            "  change eps_a after seeing this table."
        )
    return {"passed": bool(ok), "per_geometry": per_geom, "report": "\n".join(report)}


if __name__ == "__main__":
    raise SystemExit(main())
