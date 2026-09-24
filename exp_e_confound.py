"""Experiment E -- the confound (C1, C2).  The decisive experiment (§9-E).

Evaluates each trained IDM on held-out INTERACT / DECOY / ABSENT rollouts,
paired by ``(action, seed)``, and computes the object-sensitivity gap

    G = mean_p [ err(DECOY_p) - err(INTERACT_p) ]                       (§7.4)

as the **mean of per-pair differences**.  ``stats.paired_gap`` is the only route
to G in this codebase and it refuses unmatched data, so the difference-of-means
mistake (§14) is not reachable from here.

Pre-registered predictions, if the hypothesis holds:

  G(A-std) ~ 0 and err(A-std, INTERACT) ~ err(A-std, DECOY) << prior   -> C1
  err(A-del, INTERACT) << prior and near floor;
      err(A-del, DECOY) ~ prior (construction check)                   -> C2
  G(A-time) ~ G(A-std)                                                 -> T5 cleared
  G(A-clip-del) ~ G(A-std)                                             -> T8 ablation
  ABSENT quantifies the residual OOD component of any gap              -> T2

**This script computes per-condition contrasts, so it is sealed behind the
§0.5 pre-registration lock.**  It refuses to run until `prereg.md` is committed
with a timestamp earlier than these outputs and byte-identical to the committed
version.  There is no flag to skip that.

    python exp_e_confound.py [--variants A-std A-del A-time A-clip-del]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config as C
import datasets as ds
import distances as D
import exp_d_floors as expd
import idm
import idm_data as idd
import prereg_lock
import protocol as proto_mod
import provenance
import stats
import train_idm as tidm

EXPERIMENT = "exp_e_confound"


# ==========================================================================
# Evaluation of a trained checkpoint across all three conditions
# ==========================================================================


def evaluate_all_conditions(run_dir: Path, *, geometries, conditions, num_workers=4) -> dict:
    """Load a checkpoint and score held-out rollouts in every condition.

    One IDM, evaluated across conditions (§8.2).  Returns per-sample errors
    tagged with (tuple_index, geometry) so the pairing can be reconstructed and
    re-verified downstream.
    """
    ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    variant = ck["variant"]
    horizon = C.IDM_VARIANTS[variant]
    is_clip = variant in C.CLIP_VARIANTS
    is_arch_a = variant.startswith("A-")
    dev = idm.device()

    if is_arch_a:
        proto = proto_mod.load()
        model = idm.build_model(variant, proto)
    else:
        d_in = ck["state_dict"]["net.0.weight"].shape[1]
        model = idm.build_model(variant, d_in=d_in)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    batch = int(ck["cfg"]["batch_size"])
    out = {}
    for cond in conditions:
        stores = tidm.build_stores(geometries, [cond])
        if is_arch_a:
            d = (idd.ClipDataset(stores, "test", variant) if is_clip
                 else idd.PairDataset(stores, "test", horizon))
            loader = idd.make_loader(d, batch, shuffle=False, num_workers=num_workers)
            meta = d.meta()
        else:
            X, Y, meta = tidm.embed_stores(stores, "test", horizon, ck["encoder"],
                                           mask_mode=ck.get("mask_mode", "full"))
            norm = ck.get("embed_norm")
            if norm:
                X = (X - norm["mu"]) / norm["sd"]
            loader = idd.make_loader(idd.EmbeddingDataset(X, Y, meta), batch, shuffle=False)
        res = idm.evaluate_idm(model, loader, dev=dev, return_per_sample=True)
        out[cond] = {
            "per_sample_mae": res["abs_err"].mean(axis=1),
            "per_sample_mse": res["sq_err"].mean(axis=1),
            "abs_err": res["abs_err"],
            "tuple_index": meta["tuple_index"],
            "geometry": np.asarray(meta["geometry"]).astype(str),
        }
    return {"variant": variant, "seed": int(ck["seed"]), "conditions": out}


# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-root", type=Path, default=C.CHECKPOINT_ROOT)
    ap.add_argument("--variants", nargs="*",
                    default=["A-std", "A-del", "A-time", "A-clip-del",
                             "B-std", "B-del", "B-time"])
    ap.add_argument("--geometries", nargs="+", default=list(C.GEOMETRIES))
    ap.add_argument("--conditions", nargs="+", default=list(C.CONDITIONS))
    ap.add_argument("--floors", type=Path, default=C.RESULTS_ROOT / "exp_d" / "floors.json")
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "exp_e")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    # ---- §0.5 mechanical pre-registration lock.  No skip flag exists. -----
    artifacts = [args.out / "confound.npz", args.out / "results.json"]
    lock = prereg_lock.require("e", artifacts)
    print(lock.render())
    print()

    if not args.floors.exists():
        # Without the floor every C1/C2 verdict is a ratio to NaN, which
        # analyze.py would print as INCONCLUSIVE/FAIL with no hint why.
        raise SystemExit(f"{args.floors} missing: run exp_d_floors.py BEFORE Experiment E "
                         f"(C1 and C2 are ratios to the INTERACT floor)")
    floors = json.loads(args.floors.read_text())
    prior = D.prior_baseline_table(C.ACTION_RANGES)
    prior_mae = prior["pooled_mae_norm"]

    runs = expd.discover_runs(args.ckpt_root, args.variants)
    if not runs:
        raise SystemExit(f"no trained runs under {args.ckpt_root}")

    arrays: dict = {}
    results: dict = {"prior": prior, "variants": {},
                     "prereg_lock": lock.to_dict()}

    for variant, seed_dirs in sorted(runs.items()):
        print(f"--- {variant} ({len(seed_dirs)} seeds) ---", flush=True)
        gap_vals: dict = {}
        gap_ids: dict = {}
        cond_vals: dict = {c: {} for c in args.conditions}
        cond_ids: dict = {c: {} for c in args.conditions}
        geom_gap: dict = {}

        for seed, d in sorted(seed_dirs.items()):
            ev = evaluate_all_conditions(d, geometries=args.geometries,
                                         conditions=args.conditions,
                                         num_workers=args.num_workers)
            per = ev["conditions"]
            for c in args.conditions:
                arrays[f"{variant}_seed{seed}_{c}_mae"] = per[c]["per_sample_mae"]
                arrays[f"{variant}_seed{seed}_{c}_tuple"] = per[c]["tuple_index"]
                arrays[f"{variant}_seed{seed}_{c}_geometry"] = per[c]["geometry"]
                cond_vals[c][seed] = per[c]["per_sample_mae"]
                # Pair ids are (geometry, tuple_index): a bare tuple_index
                # repeats across geometries and paired_gap refuses duplicates.
                cond_ids[c][seed] = stats.pair_id(per[c]["geometry"], per[c]["tuple_index"])

            if "DECOY" in per and "INTERACT" in per:
                diffs, pair_ids = stats.paired_gap(
                    {k: per[k]["per_sample_mae"] for k in ("DECOY", "INTERACT")},
                    {k: stats.pair_id(per[k]["geometry"], per[k]["tuple_index"])
                     for k in ("DECOY", "INTERACT")},
                )
                gap_vals[seed] = diffs
                gap_ids[seed] = pair_ids
                arrays[f"{variant}_seed{seed}_G_paired"] = diffs
                arrays[f"{variant}_seed{seed}_G_pair_ids"] = pair_ids

                gsel = stats.pair_id_geometry(pair_ids)
                for g in args.geometries:
                    sel = gsel == g
                    if sel.any():
                        geom_gap.setdefault(g, {"v": {}, "i": {}})
                        geom_gap[g]["v"][seed] = diffs[sel]
                        geom_gap[g]["i"][seed] = pair_ids[sel]

        floor_cell = floors.get("variants", {}).get(variant, {}).get("pooled", {})
        floor = float(floor_cell.get("floor_mae", np.nan))

        entry: dict = {"floor_mae": floor, "conditions": {}, "G": {}}
        for c in args.conditions:
            if not cond_vals[c]:
                continue
            pd = stats.PairedData(values=cond_vals[c], pair_ids=cond_ids[c])
            s = stats.summarise(pd, n_boot=C.N_BOOTSTRAP, ci_level=C.BOOTSTRAP_CI,
                                rng_seed=C.BOOTSTRAP_SEED)
            s["over_prior"] = s["point"] / prior_mae
            s["over_floor"] = s["point"] / floor if floor == floor and floor > 0 else None
            entry["conditions"][c] = s
            print(f"  err({c:9s}) = {s['point']:.5f} "
                  f"[{s['ci_low']:.5f},{s['ci_high']:.5f}]  "
                  f"= {s['over_prior'] * 100:5.1f}% of prior")

        if gap_vals:
            pd = stats.PairedData(values=gap_vals, pair_ids=gap_ids)
            g_all = stats.summarise(pd, floor=floor if floor == floor and floor > 0 else None,
                                    n_boot=C.N_BOOTSTRAP, ci_level=C.BOOTSTRAP_CI,
                                    rng_seed=C.BOOTSTRAP_SEED)
            g_all["n_pairs"] = int(sum(len(v) for v in gap_vals.values()))
            entry["G"]["pooled"] = g_all
            print(f"  G(pooled)  = {g_all['point']:+.5f} "
                  f"[{g_all['ci_low']:+.5f},{g_all['ci_high']:+.5f}]"
                  + (f"  = {g_all['effect_over_floor']:+.3f} x floor"
                     if "effect_over_floor" in g_all else ""))
            for g, blob in geom_gap.items():
                gf = float(floors.get("variants", {}).get(variant, {})
                           .get(g, {}).get("floor_mae", np.nan))
                pdg = stats.PairedData(values=blob["v"], pair_ids=blob["i"])
                sg = stats.summarise(pdg, floor=gf if gf == gf and gf > 0 else None,
                                     n_boot=C.N_BOOTSTRAP, rng_seed=C.BOOTSTRAP_SEED)
                entry["G"][g] = sg
                print(f"    G({g:9s}) = {sg['point']:+.5f} "
                      f"[{sg['ci_low']:+.5f},{sg['ci_high']:+.5f}]")
        results["variants"][variant] = entry

    args.out.mkdir(parents=True, exist_ok=True)
    provenance.save_run(args.out, EXPERIMENT, arrays,
                        extra={"results": results, "geometries": args.geometries,
                               "conditions": args.conditions,
                               "prereg_lock": lock.to_dict()},
                        arrays_name="confound.npz")
    with open(args.out / "results.json", "w") as fh:
        json.dump(provenance._jsonable(results), fh, indent=2, sort_keys=True)

    print(f"\nwrote {args.out}")
    print("Run `python analyze.py --exp e` for the pre-registered decision table.\n"
          "This script deliberately prints no verdict: §0.3 keeps interpretation in "
          "analyze.py, reading saved arrays.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
