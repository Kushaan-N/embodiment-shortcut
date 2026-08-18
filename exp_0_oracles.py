"""Experiment 0 -- state-space oracles (§9-0).  CPU, minutes.  Gate C0.

Runs before anything touches a GPU, because it can end the project for free.

Tiny MLPs on privileged ground-truth state, one per (probe, horizon):

  (i)   arm-qpos-only  -> a, on INTERACT
  (ii)  object-pose-only -> a, on INTERACT
  (iii) arm-qpos-only  -> a, on DECOY   -- should match (i): the arm executes
        the same commanded trajectory in both conditions
  (iv)  arm *residual* (INTERACT arm - DECOY arm) -> a  -- quantifies the T9
        reaction-force channel.  If the stiff-control assertion holds, this
        probe sits at the prior baseline, i.e. the residual carries no action.
  (v)   arm+object combined -> a, on INTERACT (upper bound on state information)

What the gate means
-------------------
* (i) at s_std near the floor  ->  the shortcut channel mechanically exists (C0).
* (ii) at s_del near the PRIOR ->  the action is not recoverable from the
  settled object at all, i.e. the identifiability problem of T3 is fatal.  The
  project stops here, at a cost of zero GPU-hours, rather than after fifty
  visual IDM trainings.

Every number is reported on the [floor .. prior] axis of §7.3.  A bare error is
not interpretable and this script never prints one.

    python exp_0_oracles.py [--n 400] [--out results/exp_0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config as C
import datasets
import distances as D
import provenance
import scene
import stats

EXPERIMENT = "exp_0_oracles"

HORIZONS = ("s_0", "s_std", "s_del", "s_time")
PROBES = ("arm_only", "object_only", "arm_plus_object")


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------


def pose_features(pose: np.ndarray) -> np.ndarray:
    """7-vector pose -> 9-vector (position, first two rotation-matrix columns).

    The 6D rotation representation avoids the quaternion double cover, which a
    network fed raw (w, x, y, z) would have to learn to undo -- and would
    partly fail to, making the oracle look weaker than the state really is.
    """
    R = D.quat_to_matrix(pose[3:])
    return np.concatenate([pose[:3], R[:, 0], R[:, 1]])


def build_features(arm_h: np.ndarray, obj_h: np.ndarray | None, probe: str,
                   hi: int) -> np.ndarray:
    """Features for one rollout at horizon index ``hi``.

    Always includes the s_0 state alongside the horizon state, mirroring §6.4's
    "s_0 is the shared conditioning frame" so the oracles and the visual IDMs
    are conditioned on the same information.
    """
    arm = np.concatenate([arm_h[0], arm_h[hi]])
    if probe == "arm_only":
        return arm
    if obj_h is None:
        raise ValueError("object probe requires object poses")
    obj = np.concatenate([pose_features(obj_h[0]), pose_features(obj_h[hi])])
    if probe == "object_only":
        return obj
    return np.concatenate([arm, obj])


# --------------------------------------------------------------------------
# Tiny MLP
# --------------------------------------------------------------------------


class TinyMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int = 3, width: int = 256, depth: int = 3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_probe(Xtr, Ytr, Xva, Yva, Xte, Yte, *, seed: int, epochs: int = 400,
              lr: float = 3e-3, batch: int = 256) -> dict:
    """Train one probe; return per-sample held-out absolute errors.

    Model selection is on val MAE; the returned errors are on test.  Raw
    per-sample arrays are returned, never summary statistics (§0.3).
    """
    torch.manual_seed(seed)
    dev = "cpu"
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    xt = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32, device=dev)  # noqa: E731
    yt = lambda A: torch.tensor(A, dtype=torch.float32, device=dev)              # noqa: E731

    Xtr_t, Ytr_t = xt(Xtr), yt(Ytr)
    Xva_t, Yva_t = xt(Xva), yt(Yva)
    Xte_t = xt(Xte)

    model = TinyMLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr_t.shape[0]
    best_val, best_state = np.inf, None
    g = torch.Generator().manual_seed(seed)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xtr_t[idx]), Ytr_t[idx])
            loss.backward()
            opt.step()
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                v = float((model(Xva_t) - Yva_t).abs().mean())
            if v < best_val:
                best_val = v
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).cpu().numpy()
    return {
        "abs_err": np.abs(pred - Yte),          # (n_test, 3) normalised units
        "sq_err": (pred - Yte) ** 2,
        "pred": pred,
        "val_mae": best_val,
    }


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------


def generate(n: int, geometry: str, master_seed: int, thresholds) -> dict:
    """Simulate paired INTERACT / DECOY rollouts and keep only their states."""
    thr = thresholds.for_geometry(geometry)
    rows = {"action": [], "tuple_index": [], "split": [],
            "arm_I": [], "arm_D": [], "obj_I": [], "hz_I": [], "valid": []}
    for ti in range(n):
        streams = scene.tuple_streams(master_seed, geometry, ti)
        action = scene.sample_action(streams["action"])
        rI = scene.rollout(action, seed=ti, condition="INTERACT", geometry=geometry,
                           render=False, thresholds=thr, master_seed=master_seed)
        rD = scene.rollout(action, seed=ti, condition="DECOY", geometry=geometry,
                           render=False, thresholds=thr, master_seed=master_seed)
        hz = rI["horizon_idx"]
        idx = [hz[h] for h in HORIZONS]

        def stack(traj, traj_time, indices):
            return np.stack([
                (traj_time if (h == "s_time" and traj_time is not None) else traj)[i]
                for h, i in zip(HORIZONS, indices)
            ])

        rows["action"].append(action)
        rows["tuple_index"].append(ti)
        rows["split"].append(datasets.split_for_tuple(geometry, ti))
        rows["arm_I"].append(stack(rI["arm_qpos"], rI["arm_qpos_time"], idx))
        rows["arm_D"].append(stack(rD["arm_qpos"], rD["arm_qpos_time"], idx))
        rows["obj_I"].append(stack(rI["obj_poses"], rI["obj_poses_time"], idx))
        rows["hz_I"].append(idx)
        rows["valid"].append(rI["validators"]["_all_passed"]
                             and rD["validators"]["_all_passed"])
        if (ti + 1) % 100 == 0:
            print(f"    {geometry} {ti + 1}/{n}")
    return {k: np.asarray(v) for k, v in rows.items()}


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=400, help="(action, seed) tuples per geometry")
    ap.add_argument("--seeds", type=int, default=5, help="probe training seeds")
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "exp_0")
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args()

    thresholds = C.load_thresholds()
    lo, hi = C.action_ranges_array()
    rng_span = hi - lo
    prior = D.prior_baseline_table(C.ACTION_RANGES)
    print(f"Experiment 0 -- state-space oracles.  n={args.n}/geometry, "
          f"{args.seeds} seeds.\nprior baseline (normalised MAE) = "
          f"{prior['pooled_mae_norm']:.4f} per dim (analytic, uniform -> exactly 0.25)")

    arrays: dict = {}
    results: dict = {}

    for geometry in C.GEOMETRIES:
        print(f"  simulating {geometry} ...")
        data = generate(args.n, geometry, C.MASTER_SEED, thresholds)
        keep = data["valid"]
        print(f"    kept {int(keep.sum())}/{len(keep)} tuples after validators")

        Y = (data["action"][keep] - lo) / rng_span   # normalised to [0, 1]
        splits = data["split"][keep]
        arm_I, arm_D, obj_I = data["arm_I"][keep], data["arm_D"][keep], data["obj_I"][keep]
        arm_res = arm_I - arm_D                       # T9 residual channel

        for k, v in [("action", data["action"][keep]), ("split", splits),
                     ("arm_I", arm_I), ("arm_D", arm_D), ("obj_I", obj_I),
                     ("tuple_index", data["tuple_index"][keep])]:
            arrays[f"{geometry}_{k}"] = v

        tr, va, te = (splits == "train"), (splits == "val"), (splits == "test")
        if not (tr.any() and va.any() and te.any()):
            raise RuntimeError(f"{geometry}: a split is empty; increase --n")

        # Empirical prior baseline: best constant predictor fitted on train.
        const = Y[tr].mean(0, keepdims=True)
        prior_abs = np.abs(np.repeat(const, te.sum(), axis=0) - Y[te])
        results.setdefault(geometry, {})["prior_empirical"] = {
            "mae_per_dim": prior_abs.mean(0).tolist(),
            "mae_pooled": float(prior_abs.mean()),
            "mae_analytic_pooled": prior["pooled_mae_norm"],
        }
        arrays[f"{geometry}_prior_abs_err"] = prior_abs

        specs = []
        for hname, hidx in zip(HORIZONS, range(len(HORIZONS))):
            for probe in PROBES:
                specs.append((f"{probe}__INTERACT__{hname}", probe, arm_I, obj_I, hidx))
            specs.append((f"arm_only__DECOY__{hname}", "arm_only", arm_D, None, hidx))
            specs.append((f"arm_residual__INTERACT_minus_DECOY__{hname}",
                          "arm_only", arm_res, None, hidx))

        for label, probe, arm_src, obj_src, hidx in specs:
            X = np.stack([build_features(arm_src[i], None if obj_src is None else obj_src[i],
                                         probe, hidx) for i in range(len(Y))])
            per_seed_abs, per_seed_sq = {}, {}
            for s in range(args.seeds):
                out = fit_probe(X[tr], Y[tr], X[va], Y[va], X[te], Y[te],
                                seed=s, epochs=args.epochs)
                per_seed_abs[s] = out["abs_err"].mean(axis=1)   # per-sample, pooled dims
                per_seed_sq[s] = out["sq_err"].mean(axis=1)
                arrays[f"{geometry}_{label}_seed{s}_abs_err"] = out["abs_err"]
            pd = stats.PairedData(
                values=per_seed_abs,
                pair_ids={s: data["tuple_index"][keep][te] for s in per_seed_abs},
            )
            summ = stats.summarise(pd, n_boot=2000, rng_seed=C.BOOTSTRAP_SEED)
            summ["mae_over_prior"] = summ["point"] / prior["pooled_mae_norm"]
            results[geometry][label] = summ
            print(f"    {geometry:9s} {label:52s} MAE={summ['point']:.5f} "
                  f"[{summ['ci_low']:.5f},{summ['ci_high']:.5f}]  "
                  f"= {summ['mae_over_prior'] * 100:5.1f}% of prior")

    # ---- Gate C0 ---------------------------------------------------------
    gate = _evaluate_gate_c0(results, prior["pooled_mae_norm"])
    results["_gate_C0"] = gate
    results["_prior"] = prior

    args.out.mkdir(parents=True, exist_ok=True)
    provenance.save_run(args.out, EXPERIMENT, arrays,
                        seeds={"master": C.MASTER_SEED, "probe_seeds": list(range(args.seeds))},
                        extra={"n_per_geometry": args.n},
                        arrays_name="oracle_errors.npz")
    with open(args.out / "results.json", "w") as fh:
        json.dump(provenance._jsonable(results), fh, indent=2, sort_keys=True)

    print("\n" + "=" * 72)
    print(gate["report"])
    print("=" * 72)
    return 0 if gate["passed"] else 2


def _evaluate_gate_c0(results: dict, prior_mae: float) -> dict:
    """Gate C0, evaluated per §9-0.  Both branches must remain reachable (§0.2)."""
    # Pre-declared thresholds, stated here so they cannot be tuned after seeing
    # the numbers: "near the floor" is <= 10% of prior; "at prior" is >= 80%.
    NEAR_FLOOR = 0.10
    AT_PRIOR = 0.80

    lines, ok = [], True
    per_geom = {}
    for g in C.GEOMETRIES:
        if g not in results:
            continue
        arm_std = results[g]["arm_only__INTERACT__s_std"]["mae_over_prior"]
        obj_del = results[g]["object_only__INTERACT__s_del"]["mae_over_prior"]
        obj_std = results[g]["object_only__INTERACT__s_std"]["mae_over_prior"]
        arm_del = results[g]["arm_only__INTERACT__s_del"]["mae_over_prior"]
        arm_dec = results[g]["arm_only__DECOY__s_std"]["mae_over_prior"]
        resid = results[g]["arm_residual__INTERACT_minus_DECOY__s_std"]["mae_over_prior"]

        channel_exists = arm_std <= NEAR_FLOOR
        object_recoverable = obj_del < AT_PRIOR
        arm_matches = abs(arm_std - arm_dec) <= 0.05
        t9_clean = resid >= AT_PRIOR

        per_geom[g] = {
            "arm_only_s_std_over_prior": arm_std,
            "object_only_s_std_over_prior": obj_std,
            "arm_only_s_del_over_prior": arm_del,
            "object_only_s_del_over_prior": obj_del,
            "arm_only_DECOY_s_std_over_prior": arm_dec,
            "arm_residual_s_std_over_prior": resid,
            "C0_shortcut_channel_exists": channel_exists,
            "object_recoverable_at_s_del": object_recoverable,
            "arm_channel_condition_invariant": arm_matches,
            "T9_residual_carries_no_action": t9_clean,
        }
        ok = ok and channel_exists and object_recoverable
        lines.append(
            f"{g:9s} arm@s_std={arm_std * 100:5.1f}%prior (C0 needs <={NEAR_FLOOR * 100:.0f}%)  "
            f"obj@s_del={obj_del * 100:5.1f}%prior (needs <{AT_PRIOR * 100:.0f}%)  "
            f"arm@s_del={arm_del * 100:5.1f}%  T9resid={resid * 100:5.1f}%"
        )

    verdict = "PASS" if ok else "FAIL"
    report = [f"GATE C0: {verdict}", *lines, "", "Interpretation:"]
    if ok:
        report.append(
            "  The shortcut channel mechanically exists: arm state alone recovers the "
            "action at the standard horizon.\n  The settled object remains informative "
            "at the delayed horizon, so OG-AF has signal to measure.\n  Proceed to "
            "Experiment A/B/C."
        )
    else:
        report.append(
            "  STOP and report to a human (§0.1).  Do NOT tune a threshold, swap a "
            "probe, or resample.\n  If arm@s_std is not near the floor, C0 is refuted "
            "and the paper's premise is wrong.\n  If obj@s_del is at the prior, the "
            "identifiability problem of T3 is fatal at this horizon:\n  shorten the "
            "horizon or drop the geometry -- allowed NOW, frozen at pre-registration."
        )
    return {"passed": bool(ok), "per_geometry": per_geom,
            "thresholds": {"near_floor_frac_of_prior": NEAR_FLOOR,
                           "at_prior_frac_of_prior": AT_PRIOR},
            "report": "\n".join(report)}


if __name__ == "__main__":
    raise SystemExit(main())
