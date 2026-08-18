"""Experiment G -- Lipschitz stratification (§9-G).

Per state-action pair,

    L(s, a) = max_{a' in B(a, eps)}  ||S_gt(s, a) - S_gt(s, a')|| / ||a - a'||

estimated from ``k`` sampled perturbations.  The analytic derivative
``ds'/da`` is undefined at impulsive contact, so a sampled max is the honest
estimator -- and a max over ``k`` samples is **biased downward**, which is
reported rather than glossed.

The norm is taken in **state space with per-component normalisation** (pose
components scaled by their Experiment A delta scales), never in latent space
and never mixing raw metres with raw radians (§9-G, §14).

Budget note (§9-G): prefer fewer rollouts and more strata.  The prediction is
that the shortcut is worst in the low-sensitivity stratum -- where the object
barely responds to the action, so the arm is the only channel -- and that the
OG-AF correction matters most in the high-sensitivity stratum.

    python exp_g_lipschitz.py --n 300 --k 8
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
import stats

EXPERIMENT = "exp_g_lipschitz"


def local_lipschitz(action, seed: int, geometry: str, thr: dict, *,
                    eps: float, k: int, rng) -> dict:
    """Sampled estimate of L at one (state, action), plus the base settled state."""
    lo, hi = C.action_ranges_array()
    span = hi - lo
    base = scene.rollout(action, seed=seed, condition="INTERACT", geometry=geometry,
                         render=False, thresholds=thr)
    if not base["validators"]["_all_passed"]:
        return {"valid": False}
    p0 = base["obj_poses"][base["horizon_idx"]["s_del"]]
    p0_std = base["obj_poses"][base["horizon_idx"]["s_std"]]

    pos_scale = thr["delta_pos_min"]
    rot_scale = thr["delta_rot_min"] or thr["delta_rot_raw_min"]

    ratios, ratios_std = [], []
    for _ in range(k):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        a2 = np.clip(action + eps * span * d, lo, hi)
        da = float(np.linalg.norm((a2 - action) / span))
        if da < 1e-9:
            continue
        r2 = scene.rollout(a2, seed=seed, condition="INTERACT", geometry=geometry,
                           render=False, thresholds=thr)
        if not r2["validators"]["_all_passed"]:
            continue
        q = r2["obj_poses"][r2["horizon_idx"]["s_del"]]
        q_std = r2["obj_poses"][r2["horizon_idx"]["s_std"]]
        ds = D.state_distance(p0, q, geometry, C.GEOM_SIZE[geometry],
                              pos_scale=pos_scale, rot_scale=rot_scale, quotient=True)
        ds_std = D.state_distance(p0_std, q_std, geometry, C.GEOM_SIZE[geometry],
                                  pos_scale=pos_scale, rot_scale=rot_scale, quotient=True)
        ratios.append(ds / da)
        ratios_std.append(ds_std / da)

    if not ratios:
        return {"valid": False}
    return {"valid": True, "L_del": float(max(ratios)), "L_std": float(max(ratios_std)),
            "k_used": len(ratios), "settled_pose": p0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300, help="state-action pairs per geometry")
    ap.add_argument("--k", type=int, default=8, help="perturbations per pair")
    ap.add_argument("--eps", type=float, default=0.05,
                    help="perturbation radius, as a fraction of each action range")
    ap.add_argument("--strata", type=int, default=4)
    ap.add_argument("--geometries", nargs="+", default=list(C.GEOMETRIES))
    ap.add_argument("--gaps", type=Path, default=C.RESULTS_ROOT / "exp_e" / "confound.npz",
                    help="Experiment E per-pair G values, to stratify")
    ap.add_argument("--variant", default="A-std")
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "exp_g")
    ap.add_argument("--seed", type=int, default=C.MASTER_SEED + 23)
    args = ap.parse_args()

    thresholds = C.load_thresholds()
    rng = np.random.default_rng(args.seed)
    arrays: dict = {}
    results: dict = {"eps": args.eps, "k": args.k, "n_strata": args.strata,
                     "estimator_note": "max over k sampled perturbations; biased DOWNWARD "
                                       "as an estimate of the true local Lipschitz constant",
                     "per_geometry": {}}

    for geometry in args.geometries:
        thr = thresholds.for_geometry(geometry)
        L_del, L_std, tuples = [], [], []
        for ti in range(args.n):
            streams = scene.tuple_streams(C.MASTER_SEED, geometry, ti)
            action = scene.sample_action(streams["action"])
            out = local_lipschitz(action, ti, geometry, thr, eps=args.eps, k=args.k, rng=rng)
            if not out["valid"]:
                continue
            L_del.append(out["L_del"])
            L_std.append(out["L_std"])
            tuples.append(ti)
            if (ti + 1) % 50 == 0:
                print(f"  {geometry} {ti + 1}/{args.n}", flush=True)

        L_del = np.asarray(L_del)
        L_std = np.asarray(L_std)
        tuples = np.asarray(tuples)
        arrays[f"{geometry}_L_del"] = L_del
        arrays[f"{geometry}_L_std"] = L_std
        arrays[f"{geometry}_tuple_index"] = tuples

        edges = np.quantile(L_del, np.linspace(0, 1, args.strata + 1))
        edges[-1] += 1e-9
        strat = np.clip(np.searchsorted(edges, L_del, side="right") - 1, 0, args.strata - 1)
        arrays[f"{geometry}_stratum"] = strat

        results["per_geometry"][geometry] = {
            "n": int(L_del.size),
            "L_del_quantiles": np.quantile(L_del, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
            "L_std_quantiles": np.quantile(L_std, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
            "stratum_edges": edges.tolist(),
            "strata": [],
        }
        print(f"  {geometry}: n={L_del.size}  L_del median={np.median(L_del):.3f}  "
              f"L_std median={np.median(L_std):.3f}")

        # ---- stratified G, if Experiment E has run --------------------
        if args.gaps.exists():
            z = np.load(args.gaps, allow_pickle=False)
            gvals, gids = _collect_gaps(z, args.variant, geometry)
            if gvals is not None:
                idmap = {int(t): i for i, t in enumerate(tuples)}
                for s in range(args.strata):
                    sel_t = set(tuples[strat == s].tolist())
                    vals, ids = {}, {}
                    for seed, (v, pid) in gvals.items():
                        m = np.asarray([int(p) in sel_t for p in pid])
                        if m.any():
                            vals[seed] = v[m]
                            ids[seed] = pid[m]
                    entry = {"stratum": s, "n_pairs": int(sum(len(v) for v in vals.values())),
                             "L_range": [float(edges[s]), float(edges[s + 1])]}
                    if vals:
                        pd = stats.PairedData(values=vals, pair_ids=ids)
                        entry["G"] = stats.summarise(pd, n_boot=2000,
                                                     rng_seed=C.BOOTSTRAP_SEED)
                        print(f"    stratum {s} (L in [{edges[s]:.2f},{edges[s + 1]:.2f}]) "
                              f"G={entry['G']['point']:+.5f} "
                              f"[{entry['G']['ci_low']:+.5f},{entry['G']['ci_high']:+.5f}]")
                    results["per_geometry"][geometry]["strata"].append(entry)
        else:
            print("    (Experiment E outputs not found; stratified G skipped)")

    args.out.mkdir(parents=True, exist_ok=True)
    provenance.save_run(args.out, EXPERIMENT, arrays, seeds={"seed": args.seed},
                        extra={"results": results}, arrays_name="lipschitz.npz")
    with open(args.out / "results.json", "w") as fh:
        json.dump(provenance._jsonable(results), fh, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}")
    return 0


def _collect_gaps(z, variant: str, geometry: str):
    """Per-seed (G values, pair ids) restricted to one geometry, from Experiment E."""
    vals: dict = {}
    for key in z.files:
        if not (key.startswith(f"{variant}_seed") and key.endswith("_G_paired")):
            continue
        seed = int(key.split("_seed")[1].split("_")[0])
        ids = z[key.replace("_G_paired", "_G_pair_ids")]
        gkey = f"{variant}_seed{seed}_INTERACT_geometry"
        tkey = f"{variant}_seed{seed}_INTERACT_tuple"
        if gkey not in z.files or tkey not in z.files:
            continue
        gmap = dict(zip(z[tkey].tolist(), z[gkey].astype(str).tolist()))
        m = np.asarray([gmap.get(int(p)) == geometry for p in ids])
        if m.any():
            vals[seed] = (z[key][m], ids[m])
    return (vals, None) if vals else (None, None)


if __name__ == "__main__":
    raise SystemExit(main())
