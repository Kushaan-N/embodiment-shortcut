"""Experiment D -- identifiability floors (§9-D).  GPU.

Evaluates every trained IDM on held-out **INTERACT** rollouts and calls the
residual error the irreducible floor for that (variant, geometry) -- the T3
countermeasure that makes every downstream number interpretable.

INTERACT only, deliberately.  Per-condition contrasts -- anything that reveals
G -- stay sealed until `prereg.md` is committed and pushed (§9-D, §0.5).  This
script reads only the ``INTERACT_*`` arrays that `train_idm.py` writes by
default, and raises if asked for anything else.

Every floor is reported on the axis ``[floor .. prior]`` of §7.3.  If the s_del
floor swamps the expected signal the horizon is too lossy: shorten it and
re-run.  That is allowed NOW and frozen at pre-registration.

    python exp_d_floors.py [--variants A-std A-del ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config as C
import distances as D
import provenance
import stats

EXPERIMENT = "exp_d_floors"

#: Pre-declared: if the floor is this close to the prior, the horizon carries
#: essentially no recoverable signal and Experiment E cannot say anything.
FLOOR_SWAMPS_SIGNAL_FRAC = 0.80


def discover_runs(ckpt_root: Path, variants=None, mask_mode: str = "full",
                  augment: bool = False) -> dict:
    """Find trained runs as ``{variant: {seed: dir}}``."""
    out: dict = {}
    for d in sorted(Path(ckpt_root).glob("*__seed*")):
        meta_path = d / "metadata.json"
        if not (d / "eval.npz").exists() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        ex = meta.get("extra", {})
        if ex.get("mask_mode", "full") != mask_mode:
            continue
        if bool(ex.get("augment", False)) != augment:
            continue
        v = ex.get("variant")
        if variants and v not in variants:
            continue
        seed = int(meta.get("seeds", {}).get("train_seed", -1))
        out.setdefault(v, {})[seed] = d
    return out


def load_interact_errors(run_dir: Path) -> dict:
    """Per-sample INTERACT errors from one run.  Refuses other conditions (§9-D)."""
    z = np.load(run_dir / "eval.npz", allow_pickle=False)
    if "INTERACT_abs_err" not in z.files:
        raise RuntimeError(f"{run_dir} has no INTERACT evaluation")
    leaked = sorted({f.split("_")[0] for f in z.files
                     if f.split("_")[0] in ("DECOY", "ABSENT")})
    return {
        "abs_err": z["INTERACT_abs_err"],
        "sq_err": z["INTERACT_sq_err"],
        "tuple_index": z["INTERACT_tuple_index"],
        "geometry": z["INTERACT_geometry"].astype(str),
        "other_conditions_present": leaked,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-root", type=Path, default=C.CHECKPOINT_ROOT)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--mask-mode", default="full", choices=("full", "arm_masked", "object_masked"))
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "exp_d")
    args = ap.parse_args()

    prior = D.prior_baseline_table(C.ACTION_RANGES)
    prior_mae = prior["pooled_mae_norm"]
    prior_mse = prior["pooled_mse_norm"]

    runs = discover_runs(args.ckpt_root, args.variants, args.mask_mode, args.augment)
    if not runs:
        raise SystemExit(f"no trained runs under {args.ckpt_root} "
                         f"(mask_mode={args.mask_mode}, augment={args.augment})")

    print(f"Experiment D -- identifiability floors, INTERACT held-out only.")
    print(f"prior baseline: MAE={prior_mae:.5f}  MSE={prior_mse:.5f}  "
          f"(analytic, uniform actions)\n")

    arrays: dict = {}
    floors: dict = {"prior": prior, "variants": {}}
    warned_leak = set()

    for variant, seed_dirs in sorted(runs.items()):
        per_geom_vals: dict = {}
        per_geom_ids: dict = {}
        for seed, d in sorted(seed_dirs.items()):
            e = load_interact_errors(d)
            for c in e["other_conditions_present"]:
                if (variant, c) not in warned_leak:
                    warned_leak.add((variant, c))
                    print(f"  note: {variant} run dir also contains {c} arrays; "
                          f"Experiment D ignores them (§9-D seal)")
            per_sample = e["abs_err"].mean(axis=1)
            arrays[f"{variant}_seed{seed}_INTERACT_abs_err"] = e["abs_err"]
            arrays[f"{variant}_seed{seed}_INTERACT_geometry"] = e["geometry"]
            for g in list(C.GEOMETRIES) + ["pooled"]:
                sel = np.ones(len(per_sample), bool) if g == "pooled" else (e["geometry"] == g)
                if not sel.any():
                    continue
                per_geom_vals.setdefault(g, {})[seed] = per_sample[sel]
                per_geom_ids.setdefault(g, {})[seed] = e["tuple_index"][sel]

        floors["variants"][variant] = {}
        for g, vals in per_geom_vals.items():
            pd = stats.PairedData(values=vals, pair_ids=per_geom_ids[g])
            summ = stats.summarise(pd, n_boot=C.N_BOOTSTRAP, ci_level=C.BOOTSTRAP_CI,
                                   rng_seed=C.BOOTSTRAP_SEED)
            summ["floor_mae"] = summ["point"]
            summ["floor_over_prior"] = summ["point"] / prior_mae
            summ["swamps_signal"] = bool(summ["floor_over_prior"] >= FLOOR_SWAMPS_SIGNAL_FRAC)
            floors["variants"][variant][g] = summ
            print(f"  {variant:12s} {g:9s} floor MAE={summ['point']:.5f} "
                  f"[{summ['ci_low']:.5f},{summ['ci_high']:.5f}]  "
                  f"= {summ['floor_over_prior'] * 100:5.1f}% of prior  "
                  f"seeds[{summ['seed_min']:.5f},{summ['seed_max']:.5f}]"
                  + ("   <-- SWAMPS SIGNAL" if summ["swamps_signal"] else ""))

    swamped = [(v, g) for v, gs in floors["variants"].items()
               for g, s in gs.items() if s["swamps_signal"] and g != "pooled"]
    floors["_swamped"] = [list(x) for x in swamped]

    args.out.mkdir(parents=True, exist_ok=True)
    provenance.save_run(args.out, EXPERIMENT, arrays,
                        extra={"floors": floors, "mask_mode": args.mask_mode,
                               "augment": args.augment,
                               "runs": {v: {str(s): str(d) for s, d in sd.items()}
                                        for v, sd in runs.items()}},
                        arrays_name="floors.npz")
    with open(args.out / "floors.json", "w") as fh:
        json.dump(provenance._jsonable(floors), fh, indent=2, sort_keys=True)

    print("\n" + "=" * 72)
    if swamped:
        print("WARNING: these (variant, geometry) cells have a floor at or above "
              f"{FLOOR_SWAMPS_SIGNAL_FRAC:.0%} of the prior:")
        for v, g in swamped:
            print(f"  {v} / {g}")
        print("\n  The horizon is too lossy there and Experiment E cannot resolve an\n"
              "  effect against it.  Permitted NOW (and frozen at pre-registration):\n"
              "  shorten the delayed horizon, or drop the geometry.  Not permitted:\n"
              "  reporting E on that cell as though the floor were informative.")
    else:
        print("All floors leave headroom against the prior.\n"
              "NEXT: run power.py, then write, commit and PUSH prereg.md (§11, §3.7).\n"
              "Experiment E's analysis is sealed until the §0.5 lock passes.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
