"""Power analysis from Experiment D outputs (§9).  Feeds the pre-registration.

From D's held-out per-rollout error distribution, estimate ``Var(err)``, then
compute the ``n`` needed to detect the pre-registered minimum detectable effect.

Two deliberate conservatisms:

* **Pair correlation is assumed to be zero.**  The design pairs INTERACT with
  DECOY on ``(action, seed)``, and positive pairing correlation only shrinks
  the variance of the difference.  Assuming rho = 0 therefore over-estimates
  the required n, never under-estimates it.  The *observed* rho is reported
  alongside once E has run, but it is not used to justify a smaller corpus.

* **The MDE is a fraction of the s_std floor**, not of the prior.  The floor is
  the smaller anchor, so this is the stricter of the two readings of §9.

Output: ``n`` per geometry, and the pre-registered ``n = min(2000, computed)``.
If the computed n exceeds 2000 the corpus must be *extended before* E runs;
shards make that possible without re-simulating anything.

    python power.py [--mde-frac 0.25] [--variant A-std]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import config as C
import distances as D
import provenance

EXPERIMENT = "power_analysis"


def z_from_alpha(alpha: float, one_sided: bool = True) -> float:
    """Normal quantile without scipy (statsmodels/scipy may be absent on a login node)."""
    p = 1.0 - (alpha if one_sided else alpha / 2.0)
    # Acklam's inverse-normal approximation; accurate to ~1e-9 in this range.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def required_n(sd_diff: float, mde: float, alpha: float, power: float) -> int:
    """One-sided paired test: n = ((z_alpha + z_beta) * sd / mde)^2."""
    if mde <= 0:
        raise ValueError("minimum detectable effect must be positive")
    za = z_from_alpha(alpha, one_sided=True)
    zb = z_from_alpha(1.0 - power, one_sided=True)
    return int(math.ceil(((za + zb) * sd_diff / mde) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--floors", type=Path, default=C.RESULTS_ROOT / "exp_d" / "floors.json")
    ap.add_argument("--arrays", type=Path, default=C.RESULTS_ROOT / "exp_d" / "floors.npz")
    ap.add_argument("--variant", default="A-std",
                    help="the variant whose floor defines the MDE (C1 is about A-std)")
    ap.add_argument("--mde-frac", type=float, default=C.POWER_MDE_FRACTION_OF_FLOOR)
    ap.add_argument("--alpha", type=float, default=C.POWER_ALPHA)
    ap.add_argument("--power", type=float, default=C.POWER_TARGET)
    ap.add_argument("--rho", type=float, default=0.0,
                    help="assumed pair correlation; 0 is conservative and is the default")
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "power")
    args = ap.parse_args()

    if not args.floors.exists():
        raise SystemExit(f"{args.floors} missing; run exp_d_floors.py first")
    floors = json.loads(args.floors.read_text())
    z = np.load(args.arrays, allow_pickle=False)
    prior = D.prior_baseline_table(C.ACTION_RANGES)

    print(f"Power analysis: variant={args.variant}, MDE = {args.mde_frac:.0%} of its floor, "
          f"one-sided alpha={args.alpha}, power={args.power}, assumed rho={args.rho}")

    results = {"variant": args.variant, "mde_frac_of_floor": args.mde_frac,
               "alpha": args.alpha, "power": args.power, "assumed_rho": args.rho,
               "prior": prior, "per_geometry": {}}

    for geometry in list(C.GEOMETRIES) + ["pooled"]:
        cell = floors["variants"].get(args.variant, {}).get(geometry)
        if cell is None:
            continue
        floor = float(cell["floor_mae"])
        mde = args.mde_frac * floor

        # Per-rollout error SD, pooled over seeds, from the raw saved arrays.
        per_sample = []
        for key in z.files:
            if not key.startswith(f"{args.variant}_seed") or not key.endswith("_abs_err"):
                continue
            gkey = key.replace("_abs_err", "_geometry")
            errs = z[key].mean(axis=1)
            if geometry != "pooled":
                if gkey not in z.files:
                    continue
                errs = errs[z[gkey].astype(str) == geometry]
            per_sample.append(errs)
        if not per_sample:
            continue
        vals = np.concatenate(per_sample)
        sd = float(vals.std(ddof=1))
        # SD of the paired difference under the assumed correlation.
        sd_diff = sd * math.sqrt(2.0 * (1.0 - args.rho))
        n = required_n(sd_diff, mde, args.alpha, args.power)
        n_prereg = min(C.N_TUPLES_PER_GEOMETRY, n)
        needs_extension = n > C.N_TUPLES_PER_GEOMETRY

        results["per_geometry"][geometry] = {
            "floor_mae": floor, "mde": mde, "err_sd": sd, "sd_of_paired_diff": sd_diff,
            "n_required": n, "n_preregistered": n_prereg,
            "n_available_per_geometry": C.N_TUPLES_PER_GEOMETRY,
            "test_split_fraction": C.SPLIT_FRACTIONS[2],
            "n_test_available": int(C.N_TUPLES_PER_GEOMETRY * C.SPLIT_FRACTIONS[2]),
            "needs_corpus_extension": bool(needs_extension),
        }
        n_test = results["per_geometry"][geometry]["n_test_available"]
        flag = ""
        if n > n_test:
            flag = f"   <-- needs {n} held-out pairs, only {n_test} available"
        print(f"  {geometry:9s} floor={floor:.5f}  MDE={mde:.5f}  sd={sd:.5f}  "
              f"n_required={n:6d}  n_prereg={n_prereg:5d}{flag}")

    if not results["per_geometry"]:
        # An empty table must never read as "powered".  This happens when the
        # requested variant has no Experiment D floors -- i.e. it was never
        # trained -- and silently reporting a pass here would let an
        # underpowered corpus through into the pre-registration.
        raise SystemExit(
            f"no Experiment D floors for variant {args.variant!r} in {args.floors}. "
            f"Available: {sorted(floors.get('variants', {}))}. "
            f"Train it before running the power analysis -- an empty power table is "
            f"not a passing one."
        )

    any_ext = any(v["n_required"] > v["n_test_available"]
                  for k, v in results["per_geometry"].items() if k != "pooled")
    results["_needs_extension"] = bool(any_ext)

    args.out.mkdir(parents=True, exist_ok=True)
    provenance.save_run(args.out, EXPERIMENT, {"placeholder": np.zeros(1)},
                        extra={"results": results}, arrays_name="empty.npz")
    with open(args.out / "power.json", "w") as fh:
        json.dump(provenance._jsonable(results), fh, indent=2, sort_keys=True)

    print("\n" + "=" * 72)
    if any_ext:
        print("The corpus is UNDERPOWERED for the pre-registered MDE.\n"
              "Extend it with additional shards BEFORE writing prereg.md -- the split is\n"
              "a deterministic hash, so new tuples get assigned without re-simulating or\n"
              "re-assigning any existing tuple.  Do not lower the MDE to fit the corpus.")
    else:
        print("Powered for the pre-registered MDE.\n"
              "NEXT (§16 step 9): write prereg.md from these numbers, `git commit`, and\n"
              "PUSH to a public repository.  Experiment E's analysis stays sealed until\n"
              "prereg_lock.check() passes.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
