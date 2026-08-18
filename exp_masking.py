"""Masking decomposition (§8.4).  Architecture B only, 5 seeds, cheap, high leverage.

Using MuJoCo ground-truth segmentation, three probe families per horizon:

  full           -- the whole frame
  arm_masked     -- object pixels only  (the arm is removed)
  object_masked  -- arm pixels only     (the object is removed)

Masked regions are filled with the **background render**, not black, so a probe
cannot key on mask shape instead of the content it was denied.

Predicted orderings if the hypothesis holds:

  at s_std :  object_masked ~ full  <<  prior     and  arm_masked ~ prior
  at s_del :  the ordering flips

This is the most direct mechanistic evidence in the paper, and it reaches the
same question as the DECOY design through a completely unrelated mechanism -- so
if the two agree, T2 (object-absent is out-of-distribution) is dead: the mask
probes never remove the object from the *scene*, only from the *input*.

    python exp_masking.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import config as C
import distances as D
import exp_d_floors as expd
import idm_data as idd
import provenance
import stats

EXPERIMENT = "exp_masking"

PROBE_VARIANTS = ("B-std", "B-del", "B-time")


def train_all(variants, seeds, mask_modes, encoder, geometries, python=sys.executable):
    """Drive train_idm.py as a subprocess per cell.

    Subprocess rather than an import so each probe gets a clean process (fresh
    CUDA context, no cross-cell state), which is also how the Unity array job
    will invoke it.
    """
    for variant in variants:
        for mode in mask_modes:
            for seed in seeds:
                tag = f"{variant}__seed{seed}__{mode}"
                if (C.CHECKPOINT_ROOT / tag / "eval.npz").exists():
                    print(f"  skip {tag} (already trained)")
                    continue
                cmd = [python, str(C.REPO_ROOT / "train_idm.py"),
                       "--variant", variant, "--seed", str(seed),
                       "--mask-mode", mode, "--encoder", encoder,
                       "--geometries", *geometries]
                print(f"  train {tag}", flush=True)
                r = subprocess.run(cmd, cwd=C.REPO_ROOT)
                if r.returncode != 0:
                    raise RuntimeError(f"training failed for {tag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true", help="train the probes first")
    ap.add_argument("--seeds", type=int, default=len(C.MASKING_SEEDS))
    ap.add_argument("--variants", nargs="+", default=list(PROBE_VARIANTS))
    ap.add_argument("--mask-modes", nargs="+", default=list(idd.MASK_MODES))
    ap.add_argument("--encoder", default=C.ENCODERS[0])
    ap.add_argument("--geometries", nargs="+", default=list(C.GEOMETRIES))
    ap.add_argument("--ckpt-root", type=Path, default=C.CHECKPOINT_ROOT)
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "exp_masking")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    if args.train:
        print("training masking probes (Architecture B only, §8.4) ...")
        train_all(args.variants, seeds, args.mask_modes, args.encoder, args.geometries)

    prior = D.prior_baseline_table(C.ACTION_RANGES)
    prior_mae = prior["pooled_mae_norm"]
    arrays: dict = {}
    results: dict = {"prior": prior, "cells": {}}

    print(f"\nprior baseline MAE = {prior_mae:.5f}\n")
    for variant in args.variants:
        horizon = C.IDM_VARIANTS[variant]
        for mode in args.mask_modes:
            runs = expd.discover_runs(args.ckpt_root, [variant], mask_mode=mode)
            seed_dirs = runs.get(variant, {})
            if not seed_dirs:
                print(f"  {variant:8s} {mode:14s} -- no runs found")
                continue
            vals, ids = {}, {}
            for seed, d in sorted(seed_dirs.items()):
                e = expd.load_interact_errors(d)
                vals[seed] = e["abs_err"].mean(axis=1)
                ids[seed] = e["tuple_index"]
                arrays[f"{variant}_{mode}_seed{seed}_abs_err"] = e["abs_err"]
                arrays[f"{variant}_{mode}_seed{seed}_geometry"] = e["geometry"]
            pd = stats.PairedData(values=vals, pair_ids=ids)
            s = stats.summarise(pd, n_boot=C.N_BOOTSTRAP, rng_seed=C.BOOTSTRAP_SEED)
            s["over_prior"] = s["point"] / prior_mae
            s["horizon"] = horizon
            results["cells"][f"{variant}|{mode}"] = s
            print(f"  {variant:8s} ({horizon:6s}) {mode:14s} MAE={s['point']:.5f} "
                  f"[{s['ci_low']:.5f},{s['ci_high']:.5f}] = "
                  f"{s['over_prior'] * 100:5.1f}% of prior")

    results["ordering_checks"] = _check_orderings(results["cells"])

    args.out.mkdir(parents=True, exist_ok=True)
    provenance.save_run(args.out, EXPERIMENT, arrays,
                        seeds={"probe_seeds": seeds},
                        extra={"results": results, "encoder": args.encoder},
                        arrays_name="masking.npz")
    with open(args.out / "results.json", "w") as fh:
        json.dump(provenance._jsonable(results), fh, indent=2, sort_keys=True)

    print("\n--- predicted orderings (§8.4) ---")
    for k, v in results["ordering_checks"].items():
        print(f"  {k:44s} {'HOLDS' if v['holds'] else 'does NOT hold'}   {v['detail']}")
    print(f"\nwrote {args.out}.  Interpretation: analyze.py --exp masking.")
    return 0


def _check_orderings(cells: dict) -> dict:
    """Test the two orderings §8.4 predicts, without editorialising."""
    out = {}

    def frac(variant, mode):
        c = cells.get(f"{variant}|{mode}")
        return None if c is None else c["over_prior"]

    for variant, label in (("B-std", "s_std"), ("B-del", "s_del"), ("B-time", "s_time")):
        full = frac(variant, "full")
        arm_px = frac(variant, "object_masked")   # arm pixels only
        obj_px = frac(variant, "arm_masked")      # object pixels only
        if None in (full, arm_px, obj_px):
            continue
        if label == "s_std":
            holds = (arm_px < 0.25) and (abs(arm_px - full) < 0.15) and (obj_px > 0.60)
            pred = "arm-pixels ~ full << prior, object-pixels ~ prior"
        else:
            holds = (obj_px < 0.60) and (arm_px > 0.80)
            pred = "object-pixels informative, arm-pixels ~ prior"
        out[f"{label}: {pred}"] = {
            "holds": bool(holds),
            "detail": (f"full={full * 100:.1f}%  arm_px={arm_px * 100:.1f}%  "
                       f"obj_px={obj_px * 100:.1f}%  (of prior)"),
            "full_over_prior": full, "arm_pixels_over_prior": arm_px,
            "object_pixels_over_prior": obj_px,
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
