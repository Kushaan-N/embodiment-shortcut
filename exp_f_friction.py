"""Experiment F -- physics counterfactual, dose-response (§9-F).  Confirmatory.

Identical commanded arm trajectories; the object's **sliding friction** is swept
over five log-spaced multipliers.  Not a binary flip, and deliberately not mass:
mass is visually unobservable, so an IDM that ignores it is *correctly*
ignorant and penalising that would be a bad metric, not a good result.

Prediction: delayed-horizon error responds monotonically to friction magnitude
while standard-horizon error stays flat.

Framing defence, to state in the paper rather than hide: the evaluation IDMs
never trained on varied friction **by design**.  Detecting off-distribution
dynamics is the metric's job -- a world model with wrong physics is precisely an
OOD-dynamics generator.  The ablation arm (`--include-randomised`) additionally
scores a friction-randomised IDM variant, which separates "sensitive to
friction" from merely "brittle to friction".

Note the per-geometry caveat the design forces us to report: the sphere's
settled position is governed by *rolling* friction, so a sliding-friction sweep
moves it much less.  That is a real per-geometry result (T6), not a bug, and it
is reported rather than smoothed over.

    python exp_f_friction.py --generate      # simulate the swept corpora
    python exp_f_friction.py                 # score them with trained IDMs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config as C
import datasets as ds
import distances as D
import exp_d_floors as expd
import exp_e_confound as expe
import idm_data as idd
import provenance
import stats

EXPERIMENT = "exp_f_friction"


def generate(geometries, multipliers, n_tuples, shard_size, clips: bool):
    thresholds = C.load_thresholds()
    for g in geometries:
        for fm in multipliers:
            if abs(fm - 1.0) < 1e-9:
                print(f"  {g} x{fm:g}: baseline corpus already exists, skipping")
                continue
            n_shards = int(np.ceil(n_tuples / shard_size))
            for s in range(n_shards):
                info = ds.build_shard(
                    g, s, conditions=("INTERACT",), shard_size=shard_size,
                    friction_mult=fm, thresholds=thresholds,
                    clip_root=C.CLIP_ROOT if clips else None,
                )
                print(f"  {g} x{fm:g} shard {s}: {list(info['written'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--geometries", nargs="+", default=list(C.GEOMETRIES))
    ap.add_argument("--multipliers", type=float, nargs="+",
                    default=list(C.FRICTION_MULTIPLIERS))
    ap.add_argument("--n-tuples", type=int, default=400)
    ap.add_argument("--shard-size", type=int, default=C.SHARD_SIZE)
    ap.add_argument("--clips", action="store_true")
    ap.add_argument("--variants", nargs="*", default=["A-std", "A-del"])
    ap.add_argument("--ckpt-root", type=Path, default=C.CHECKPOINT_ROOT)
    ap.add_argument("--floors", type=Path, default=C.RESULTS_ROOT / "exp_d" / "floors.json")
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "exp_f")
    args = ap.parse_args()

    if args.generate:
        print("generating friction-swept INTERACT corpora ...")
        generate(args.geometries, args.multipliers, args.n_tuples, args.shard_size,
                 args.clips)
        print("materialising frame stores ...")
        for g in args.geometries:
            for fm in args.multipliers:
                idd.materialise_frames(g, "INTERACT", friction_mult=fm)
        print("done; re-run without --generate to score.")
        return 0

    floors = json.loads(args.floors.read_text()) if args.floors.exists() else {"variants": {}}
    prior = D.prior_baseline_table(C.ACTION_RANGES)
    runs = expd.discover_runs(args.ckpt_root, args.variants)
    if not runs:
        raise SystemExit(f"no trained runs under {args.ckpt_root}")

    arrays: dict = {}
    results: dict = {"multipliers": args.multipliers, "prior": prior, "variants": {}}

    for variant, seed_dirs in sorted(runs.items()):
        print(f"--- {variant} ---", flush=True)
        per_mult: dict = {}
        for fm in args.multipliers:
            vals, ids = {}, {}
            for seed, d in sorted(seed_dirs.items()):
                # Reuse Experiment E's checkpoint evaluator, pointed at the
                # friction-swept stores.  Same model, same held-out actions;
                # only the physics differs.
                ev = _evaluate_at_friction(d, args.geometries, fm)
                vals[seed] = ev["per_sample_mae"]
                ids[seed] = ev["tuple_index"]
                arrays[f"{variant}_fm{fm:g}_seed{seed}_mae"] = ev["per_sample_mae"]
                arrays[f"{variant}_fm{fm:g}_seed{seed}_geometry"] = ev["geometry"]
            pd = stats.PairedData(values=vals, pair_ids=ids)
            floor = float(floors.get("variants", {}).get(variant, {})
                          .get("pooled", {}).get("floor_mae", np.nan))
            s = stats.summarise(pd, floor=floor if floor == floor and floor > 0 else None,
                                n_boot=C.N_BOOTSTRAP, rng_seed=C.BOOTSTRAP_SEED)
            s["over_prior"] = s["point"] / prior["pooled_mae_norm"]
            per_mult[f"{fm:g}"] = s
            print(f"  friction x{fm:<5g} MAE={s['point']:.5f} "
                  f"[{s['ci_low']:.5f},{s['ci_high']:.5f}]  "
                  f"= {s['over_prior'] * 100:5.1f}% of prior")

        xs = np.log(np.asarray(args.multipliers, dtype=float))
        ys = np.asarray([per_mult[f"{m:g}"]["point"] for m in args.multipliers])
        slope = float(np.polyfit(xs, ys, 1)[0])
        rho = float(_spearman(xs, ys))
        results["variants"][variant] = {
            "per_multiplier": per_mult,
            "log_friction_slope": slope,
            "spearman_rho": rho,
            "monotone": bool(abs(rho) > 0.9),
        }
        print(f"  dose-response: slope(d MAE / d log friction)={slope:+.5f}  "
              f"spearman={rho:+.3f}  {'monotone' if abs(rho) > 0.9 else 'NOT monotone'}")

    args.out.mkdir(parents=True, exist_ok=True)
    provenance.save_run(args.out, EXPERIMENT, arrays, extra={"results": results},
                        arrays_name="friction.npz")
    with open(args.out / "results.json", "w") as fh:
        json.dump(provenance._jsonable(results), fh, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}.  Interpretation lives in analyze.py --exp f.")
    return 0


def _evaluate_at_friction(run_dir: Path, geometries, friction_mult: float) -> dict:
    """Score one checkpoint on the friction-swept INTERACT corpus."""
    import torch

    import idm
    import protocol as proto_mod
    import train_idm as tidm

    ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    variant = ck["variant"]
    horizon = C.IDM_VARIANTS[variant]
    is_clip = variant in C.CLIP_VARIANTS
    is_arch_a = variant.startswith("A-")

    if is_arch_a:
        model = idm.build_model(variant, proto_mod.load())
    else:
        model = idm.build_model(variant, d_in=ck["state_dict"]["net.0.weight"].shape[1])
    model.load_state_dict(ck["state_dict"])

    stores = [idd.FrameStore(g, "INTERACT", friction_mult=friction_mult) for g in geometries]
    batch = int(ck["cfg"]["batch_size"])
    if is_arch_a:
        d = (idd.ClipDataset(stores, "test", variant) if is_clip
             else idd.PairDataset(stores, "test", horizon))
        loader = idd.make_loader(d, batch, shuffle=False)
        meta = d.meta()
    else:
        X, Y, meta = tidm.embed_stores(stores, "test", horizon, ck["encoder"])
        norm = ck.get("embed_norm")
        if norm:
            X = (X - norm["mu"]) / norm["sd"]
        loader = idd.make_loader(idd.EmbeddingDataset(X, Y, meta), batch, shuffle=False)
    res = idm.evaluate_idm(model, loader, return_per_sample=True)
    return {"per_sample_mae": res["abs_err"].mean(axis=1),
            "tuple_index": meta["tuple_index"],
            "geometry": np.asarray(meta["geometry"]).astype(str)}


def _spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
