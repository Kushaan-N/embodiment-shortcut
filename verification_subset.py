"""Verification-subset audit -- what a manipulator-only subset costs (§9-0b).

Descriptive, never the basis of a confirmatory claim.  Zero new simulation and
zero GPU: everything here is recomputed from the arrays Experiment 0 already
wrote.

Why this exists
---------------
The embodiment shortcut is not only an accident of how the metric is
implemented -- recent work *specifies* it.  World Action Verifier
(arXiv:2604.01985) builds a sparse inverse model restricted to a "verification
subset" S and proves (Proposition 3.1) that such a model recovers correct
actions out-of-support provided:

  (i)   z_S^{t+1} depends only on (z_S^t, a^t) and *not on the rest of the
        scene*;
  (ii)  (z_S^t, a^t) stays on-support even when the full transition is OOS;
  (iii) the action is identifiable from the subset transition
        (z_S^t, z_S^{t+1}).

Their own reading of S is "the agent's own motion pattern (e.g. joint-angle
trajectories)".  That is exactly the manipulator channel this project measures.
The paper treats (i)-(iii) as a robustness guarantee.  For *verification* it is
one.  For *evaluation* it is the failure mode: condition (i) says in so many
words that the subset carries no information about the object, so any
action-following score computed from it is blind to contact physics by
construction, not by accident.

This module makes that trade-off a measurement rather than a rhetorical point.
Conditions (i) and (iii) are directly observable in this testbed; (ii) is not
(it needs out-of-support data this corpus does not contain) and is reported as
such rather than estimated.

    python verification_subset.py [--exp0 results/exp_0] [--out results/verification_subset]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config as C
import provenance

EXPERIMENT = "verification_subset"

HORIZONS = ("s_0", "s_std", "s_del", "s_time")

#: Arm DOF units, for reporting condition (i) in physical rather than raw terms.
ARM_DOF_UNITS = ("m", "m", "m", "rad")

#: "Carries the action" and "carries nothing", on the [floor .. prior] axis of
#: §7.3.  Same numbers Experiment 0's gate uses, quoted here so the two tables
#: cannot drift apart.
NEAR_FLOOR_FRAC = 0.10
AT_PRIOR_FRAC = 0.80


def condition_i(arm_I: np.ndarray, arm_D: np.ndarray) -> dict:
    """Scene-independence of the subset: does the arm move identically?

    arm_I, arm_D: (N, n_horizons, n_dof).  Condition (i) of Proposition 3.1
    holds exactly when these agree -- the subset's evolution cannot depend on
    the rest of the scene if it is bit-for-bit the same whether or not the
    object was ever in the way.
    """
    out = {}
    for hi, h in enumerate(HORIZONS):
        d = np.abs(arm_I[:, hi] - arm_D[:, hi])
        out[h] = {
            "max_abs_dev_per_dof": d.max(axis=0).tolist(),
            "mean_abs_dev_per_dof": d.mean(axis=0).tolist(),
            "max_abs_dev": float(d.max()),
            "bitwise_identical": bool(np.array_equal(arm_I[:, hi], arm_D[:, hi])),
            "dof_units": list(ARM_DOF_UNITS),
        }
    return out


def audit_geometry(res_g: dict, arrays, geometry: str) -> dict:
    """One geometry: conditions (i) and (iii), and what the subset discards."""
    arm_I = arrays[f"{geometry}_arm_I"]
    arm_D = arrays[f"{geometry}_arm_D"]
    ci = condition_i(arm_I, arm_D)

    per_horizon = {}
    for h in HORIZONS:
        arm_key = f"arm_only__INTERACT__{h}"
        obj_key = f"object_only__INTERACT__{h}"
        both_key = f"arm_plus_object__INTERACT__{h}"
        arm_over_prior = res_g[arm_key]["mae_over_prior"]
        obj_over_prior = res_g[obj_key]["mae_over_prior"]
        both_over_prior = res_g[both_key]["mae_over_prior"]

        per_horizon[h] = {
            # (i)
            "subset_scene_independent": ci[h]["bitwise_identical"],
            "max_abs_arm_deviation": ci[h]["max_abs_dev"],
            # (iii)
            "subset_recovers_action_over_prior": arm_over_prior,
            "condition_iii_holds": bool(arm_over_prior <= NEAR_FLOOR_FRAC),
            # what choosing this subset throws away
            "object_only_over_prior": obj_over_prior,
            "arm_plus_object_over_prior": both_over_prior,
            "object_adds_over_subset": float(arm_over_prior - both_over_prior),
            "discarded_object_signal": float(max(0.0, 1.0 - obj_over_prior)),
        }
    return {"condition_i_detail": ci, "per_horizon": per_horizon}


def summarise(audit: dict) -> dict:
    """The one sentence this module exists to license, as numbers."""
    rows, both_hold = [], {}
    for g, a in audit.items():
        if g.startswith("_"):
            continue
        for h in HORIZONS:
            p = a["per_horizon"][h]
            holds = bool(p["subset_scene_independent"] and p["condition_iii_holds"])
            both_hold.setdefault(h, []).append(holds)
            rows.append({
                "geometry": g, "horizon": h,
                "cond_i": p["subset_scene_independent"],
                "cond_iii": p["condition_iii_holds"],
                "subset_over_prior": p["subset_recovers_action_over_prior"],
                "object_only_over_prior": p["object_only_over_prior"],
                "both_conditions_hold": holds,
            })
    return {
        "rows": rows,
        "horizons_where_both_hold": sorted(h for h, v in both_hold.items() if all(v)),
        "condition_ii_status": (
            "NOT MEASURABLE in this corpus: (ii) concerns out-of-support "
            "transitions, which this corpus does not contain by construction. "
            "Reported as unmeasured rather than estimated."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp0", type=Path, default=C.RESULTS_ROOT / "exp_0")
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / EXPERIMENT)
    args = ap.parse_args()

    res_path = args.exp0 / "results.json"
    arr_path = args.exp0 / "oracle_errors.npz"
    if not res_path.exists() or not arr_path.exists():
        print(f"missing {res_path} or {arr_path}; run exp_0_oracles.py first")
        return 1

    exp0 = json.loads(res_path.read_text())
    arrays = np.load(arr_path, allow_pickle=False)

    audit = {}
    for g in C.GEOMETRIES:
        if g not in exp0:
            continue
        audit[g] = audit_geometry(exp0[g], arrays, g)
    if not audit:
        print("no geometries found in Experiment 0 results")
        return 1

    audit["_summary"] = summarise(audit)

    print(f"Verification-subset audit (arXiv:2604.01985 Proposition 3.1 conditions)\n"
          f"subset S = the manipulator channel; near-floor <= {NEAR_FLOOR_FRAC:.0%} of prior\n")
    print(f"{'geometry':10s} {'horizon':8s} {'(i) scene-indep':>16s} "
          f"{'(iii) subset':>13s} {'object-only':>12s}")
    for r in audit["_summary"]["rows"]:
        print(f"{r['geometry']:10s} {r['horizon']:8s} "
              f"{str(r['cond_i']):>16s} "
              f"{r['subset_over_prior'] * 100:12.1f}% "
              f"{r['object_only_over_prior'] * 100:11.1f}%")

    both = audit["_summary"]["horizons_where_both_hold"]
    print(f"\nBoth (i) and (iii) hold at: {', '.join(both) if both else 'no horizon'}")
    if both:
        print(
            "  At these horizons a manipulator-only verification subset satisfies\n"
            "  Proposition 3.1 (i) and (iii) -- and condition (i) is precisely the\n"
            "  statement that the subset carries no object information.  An\n"
            "  action-following score computed from it therefore has G = 0 by\n"
            "  construction: it cannot distinguish correct contact physics from\n"
            "  none at all.  This is the cost of the guarantee, not a defect in\n"
            "  the proof."
        )
    print(f"\n(ii) {audit['_summary']['condition_ii_status']}")

    args.out.mkdir(parents=True, exist_ok=True)
    provenance.save_run(args.out, EXPERIMENT, {"dummy": np.array([1])},
                        extra={"near_floor_frac": NEAR_FLOOR_FRAC,
                               "at_prior_frac": AT_PRIOR_FRAC,
                               "source_exp0": str(args.exp0),
                               "reference": "arXiv:2604.01985 Proposition 3.1"},
                        arrays_name="empty.npz")
    with open(args.out / "results.json", "w") as fh:
        json.dump(provenance._jsonable(audit), fh, indent=2, sort_keys=True)
    print(f"\nwrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
