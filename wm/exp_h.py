"""Experiment H -- world-model ladder (C3) (§9-H).

Do not build until 0-G pass human review.

Scores every model on the ladder, over the same held-out action set, under

  (a) standard-horizon IDM error  (A-std, augmented)
  (b) OG-AF                       (A-del, augmented)
  (c) DINO state distance at s_del

and answers:

  **C3(a) within-model** -- Spearman correlation between the per-rollout
  rankings induced by (a) and (b), for WM-base.  Well below 1 means the
  correction changes which rollouts you think are good.

  **C3(b) between-model** -- WM-physics-corrupted vs WM-base separation under
  each metric.  Prediction: indistinguishable under (a), clearly separated
  under (b)/(c).  This is the sentence that gets the paper accepted.

  **Ladder verification** -- privileged ground-truth rollout error must be
  monotone across models 1-5.  If it is not, the ladder failed and the
  between-model claims are void.  Checked first, and reported either way.

Sealed behind the §0.5 pre-registration lock, like Experiment E.

    python wm/exp_h.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C  # noqa: E402
import distances as D  # noqa: E402
import idm  # noqa: E402
import prereg_lock  # noqa: E402
import protocol as proto_mod  # noqa: E402
import provenance  # noqa: E402
from wm.wm_generate import generated_subdir  # noqa: E402

EXPERIMENT = "exp_h_world_model"
LADDER_PATH = Path(__file__).resolve().parent / "ladder.yaml"


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    s = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / max(n1 + n2 - 2, 1))
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def paired_effect(a_scores, a_ids, b_scores, b_ids, *, n_boot: int = 10_000,
                  rng_seed: int = C.BOOTSTRAP_SEED) -> dict:
    """Paired between-model effect on the SAME held-out actions.

    Every ladder model is generated from the same held-out (action, seed)
    tuples (§9-H), so the natural estimator is the per-tuple difference
    b - a, not two independent samples.  d_paired = mean(diff) / sd(diff),
    with a percentile bootstrap CI over tuples (seeded, deterministic).
    Reported ALONGSIDE the registered unpaired Cohen's d (DEVIATIONS.md).
    """
    a_map = {int(t): float(s) for t, s in zip(a_ids, a_scores)}
    b_map = {int(t): float(s) for t, s in zip(b_ids, b_scores)}
    common = sorted(set(a_map) & set(b_map))
    if len(common) < 3:
        return {"n_pairs": len(common), "note": "fewer than 3 matched tuples"}
    diff = np.array([b_map[t] - a_map[t] for t in common], dtype=np.float64)
    sd = diff.std(ddof=1)
    d = float(diff.mean() / sd) if sd > 0 else 0.0
    rng = np.random.default_rng(rng_seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        s = diff[rng.integers(0, len(diff), size=len(diff))]
        ssd = s.std(ddof=1)
        boot[i] = s.mean() / ssd if ssd > 0 else 0.0
    return {"n_pairs": int(len(common)), "mean_diff": float(diff.mean()),
            "d_paired": d, "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)),
            "unmatched": int(len(a_map) + len(b_map) - 2 * len(common))}


def load_generated(model_name: str, geometry: str, root: Path) -> dict:
    d = Path(root) / model_name / geometry
    items = sorted(d.glob("*.npz"))
    if not items:
        raise FileNotFoundError(f"no generated rollouts under {d}")
    # Content gate (§13.5): the generator writes every item to disk BEFORE its
    # per-item validation, so the analysis must consult the shard manifests
    # and drop what failed -- otherwise "validate content" never gated anything.
    manifests = sorted(d.glob("manifest_shard*.json"))
    if not manifests:
        raise FileNotFoundError(f"{d}: no manifest_shard*.json -- generation did not finish "
                                f"(or aborted); not scoring unvalidated items")
    status: dict = {}
    for m in manifests:
        for it in json.loads(m.read_text())["items"]:
            status[it["id"]] = bool(it["passed"])
    unlisted = [p.stem for p in items if p.stem not in status]
    if unlisted:
        raise FileNotFoundError(f"{d}: {len(unlisted)} item(s) on disk are in no manifest "
                                f"(e.g. {unlisted[:3]}); re-run generation for this model")
    failed = [p for p in items if not status[p.stem]]
    if failed:
        print(f"  {model_name}/{geometry}: dropping {len(failed)}/{len(items)} items that "
              f"failed per-item validation", flush=True)
    items = [p for p in items if status[p.stem]]
    gen, cond, act, real, ids, fidx = [], [], [], [], [], []
    for p in items:
        with np.load(p) as z:
            gen.append(z["generated"])
            cond.append(z["conditioning"])
            act.append(z["action"])
            real.append(z["real_target"])
            if "frame_indices" in z.files:
                fidx.append(z["frame_indices"].astype(np.int64))
        ids.append(int(p.stem))
    return {"generated": np.stack(gen), "conditioning": np.stack(cond),
            "action": np.stack(act), "real": np.stack(real),
            "tuple_index": np.asarray(ids),
            "frame_indices": np.stack(fidx) if len(fidx) == len(items) else None}


@torch.no_grad()
def score_with_idm(variant: str, seed: int, batch: np.ndarray, actions: np.ndarray,
                   *, augmented: bool, ckpt_root: Path, device) -> np.ndarray:
    """Per-rollout normalised MAE from a trained IDM on (s_0, generated) pairs."""
    tag = f"{variant}__seed{seed}__full" + ("__aug" if augmented else "")
    ck_path = Path(ckpt_root) / tag / "checkpoint.pt"
    if not ck_path.exists():
        raise SystemExit(
            f"{ck_path} missing.  ladder.yaml scores the standard/OG-AF metrics with "
            f"AUGMENTED IDMs (evaluation.metrics.augmented: true) and also reports the "
            f"clean ones; train it with `train_idm.py --variant {variant} --seed {seed}"
            f"{' --augment' if augmented else ''}` (unity/train_idm.sbatch array items "
            f"20-29 are the augmented A-std/A-del seeds)")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = idm.build_model(variant, proto_mod.load())
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()

    y = idm.normalise_action(actions)
    errs = []
    for i in range(0, len(batch), 32):
        x = torch.from_numpy(np.ascontiguousarray(batch[i : i + 32]))
        x = x.permute(0, 1, 4, 2, 3).contiguous().float().to(device) / 255.0
        p = model(x).float().cpu().numpy()
        errs.append(np.abs(p - y[i : i + 32]).mean(axis=1))
    return np.concatenate(errs)


@torch.no_grad()
def dino_state_distance(generated: np.ndarray, real: np.ndarray, encoder_name: str,
                        device) -> np.ndarray:
    import exp_b_resolution as expb

    enc = expb.FrozenEncoder(encoder_name, device)
    a = enc(generated)["cls"]
    b = enc(real)["cls"]
    return expb.cosine_distance(a, b)


OBJECT_MASK_THRESHOLD = 20   # grey levels; prereg_addendum_H2.md §4


def object_region_error(generated: np.ndarray, real: np.ndarray, absent: np.ndarray,
                        thr: int = OBJECT_MASK_THRESHOLD) -> np.ndarray:
    """Per-item error on the OBJECT's pixels only (prereg_addendum_H2.md §4).

    Mask M_t = pixels where max_channel |real_t - absent_t| > thr, i.e. where
    the real INTERACT frame differs from the same-tuple ABSENT frame (same arm,
    no object): the object and its shadow.  Error = sum |gen - real| over M,
    normalised by 3 * |M| * 255.  Whole-clip pixel error is dominated by the
    arm and background (the object is ~2 % of the frame), which is why
    Experiment H's ladder gate could not see a physics corruption.
    """
    out = np.empty(len(generated), dtype=np.float64)
    for i in range(len(generated)):
        r = real[i].astype(np.int16)
        m = (np.abs(r - absent[i].astype(np.int16)).max(axis=-1) > thr)        # (T, H, W)
        n = int(m.sum())
        if n == 0:
            out[i] = np.nan
            continue
        d = np.abs(generated[i].astype(np.int16) - r)                          # (T, H, W, 3)
        out[i] = float(d[m].sum()) / (3.0 * n * 255.0)
    return out


def load_real_clips(tuple_ids, geometry: str, condition: str, clip_root: Path) -> np.ndarray:
    """Real WM clips (same renderer and time grid as training) for these tuples."""
    clips = []
    for t in tuple_ids:
        p = Path(clip_root) / geometry / condition / f"{int(t):07d}.npz"
        with np.load(p) as z:
            clips.append(z["clip"])
    return np.stack(clips)


def paired_contrast(std_a, std_b, og_a, og_b, ids_a, ids_b, *, n_boot: int = 10_000,
                    rng_seed: int = C.BOOTSTRAP_SEED) -> dict:
    """prereg_addendum_H2.md §5: paired d for the standard metric and OG-AF,
    and their contrast, on the SAME bootstrap resamples of tuples."""
    a_s = {int(t): float(v) for t, v in zip(ids_a, std_a)}
    b_s = {int(t): float(v) for t, v in zip(ids_b, std_b)}
    a_o = {int(t): float(v) for t, v in zip(ids_a, og_a)}
    b_o = {int(t): float(v) for t, v in zip(ids_b, og_b)}
    common = sorted(set(a_s) & set(b_s))
    ds = np.array([b_s[t] - a_s[t] for t in common])
    do = np.array([b_o[t] - a_o[t] for t in common])

    def d(x):
        sd = x.std(ddof=1)
        return float(x.mean() / sd) if sd > 0 else 0.0

    rng = np.random.default_rng(rng_seed)
    bs, bo = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(common), size=len(common))
        bs[i], bo[i] = d(ds[idx]), d(do[idx])
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    return {"n_pairs": len(common), "d_std": d(ds), "d_std_ci": q(bs),
            "d_ogaf": d(do), "d_ogaf_ci": q(bo),
            "delta": d(do) - d(ds), "delta_ci": q(bo - bs)}


def ground_truth_error(generated: np.ndarray, real: np.ndarray) -> np.ndarray:
    """Privileged comparison against the simulator's own video.

    Averaged over the whole clip, not just the final frame: a model that
    happens to land the settled pose after diverging wildly mid-rollout is not
    a good world model, and the ladder-monotonicity check depends on this
    number ordering the models honestly.
    """
    # Per item, in float32: materialising float64 copies of the whole
    # (N, 16, 224, 224, 3) arrays is ~116 GB at N=1500 (OOM on --mem=64G).
    out = np.empty(len(generated), dtype=np.float64)
    for i in range(len(generated)):
        out[i] = float(np.abs(generated[i].astype(np.float32)
                              - real[i].astype(np.float32)).mean()) / 255.0
    return out


def score_generated(name: str, g: dict, args, dev) -> dict:
    """Score one set of generated rollouts with every metric of §9-H."""
    gen = g["generated"]
    if gen.ndim != 5 or g.get("frame_indices") is None:
        raise SystemExit(f"{name}: generated items carry no frame_indices; regenerate "
                         f"with the current wm/wm_generate.py")
    gen_last = gen[:, -1]
    k_std = np.argmin(np.abs(g["frame_indices"] * C.FRAME_STRIDE
                             - C.PHASES.push_end_idx), axis=1)
    gen_std = gen[np.arange(len(gen)), k_std]
    pairs_std = np.stack([g["conditioning"][:, 0], gen_std], axis=1)
    pairs_del = np.stack([g["conditioning"][:, 0], gen_last], axis=1)
    print(f"  {name}: standard metric scored at clip frame(s) "
          f"{sorted(set(k_std.tolist()))} (nearest s_std={C.PHASES.push_end_idx}), "
          f"OG-AF at the last frame", flush=True)

    std_e = np.mean([score_with_idm("A-std", s, pairs_std, g["action"], augmented=True,
                                    ckpt_root=args.ckpt_root, device=dev)
                     for s in args.seeds], axis=0)
    del_e = np.mean([score_with_idm("A-del", s, pairs_del, g["action"], augmented=True,
                                    ckpt_root=args.ckpt_root, device=dev)
                     for s in args.seeds], axis=0)
    std_clean = np.mean([score_with_idm("A-std", s, pairs_std, g["action"],
                                        augmented=False, ckpt_root=args.ckpt_root,
                                        device=dev)
                         for s in args.seeds], axis=0)
    real_last = g["real"][:, -1] if g["real"].ndim == 5 else g["real"]
    dino = dino_state_distance(gen_last, real_last, args.encoder, dev)
    gt = ground_truth_error(g["generated"], g["real"])

    print(f"  {name:24s} standard={std_e.mean():.5f}  ogaf={del_e.mean():.5f}  "
          f"dino={dino.mean():.5f}  gt={gt.mean():.5f}")
    for k, v in (("standard", std_e), ("ogaf", del_e), ("standard_clean", std_clean),
                 ("dino", dino), ("ground_truth", gt)):
        if not np.isfinite(v).all():
            # cohens_d / paired_effect would turn a NaN into "no effect" (d = 0).
            raise RuntimeError(f"{name}: non-finite {k} scores ({int((~np.isfinite(v)).sum())})")
    return {"standard": std_e, "ogaf": del_e, "standard_clean": std_clean,
            "dino": dino, "ground_truth": gt}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generated-root", type=Path, default=C.DATA_ROOT / "generated")
    ap.add_argument("--ckpt-root", type=Path, default=C.CHECKPOINT_ROOT)
    ap.add_argument("--geometry", default="box", choices=C.GEOMETRIES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--encoder", default=C.GATE_B_SELECTED_ENCODER)
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "exp_h")
    ap.add_argument("--ladder", type=Path, default=LADDER_PATH)
    ap.add_argument("--clips", type=Path, default=C.DATA_ROOT / "wm_clips",
                    help="real WM clips (INTERACT + ABSENT) for the object-region ground truth")
    args = ap.parse_args()

    lock = prereg_lock.require("h", [args.out / "results.json"])
    print(lock.render(), "\n")

    ladder = yaml.safe_load(Path(args.ladder).read_text())
    gate_gt = ladder.get("ladder_gt", "whole_clip")
    c3b = ladder.get("c3b", {"base": "WM-base-100", "corrupted": "WM-physics-corrupted"})
    if ladder.get("addendum"):
        # A ladder registered by an addendum is sealed by THAT document's lock.
        add = C.REPO_ROOT / ladder["addendum"]
        cal = C.RESULTS_ROOT / "exp_h2_calibration" / "results.json"
        rep = prereg_lock.check([args.out / "results.json", cal], prereg_path=add)
        print(rep.render(), "\n")
        if not rep.passed:
            raise prereg_lock.PreregViolation(f"{add.name} lock failed; H2 is sealed")
    names = [m["name"] for m in ladder["models"]]
    dev = idm.device()

    arrays: dict = {}
    per_model: dict = {}
    for name in names:
        try:
            g = load_generated(name, args.geometry, args.generated_root)
        except FileNotFoundError as exc:
            print(f"  {name}: {exc}")
            continue
        # Each IDM scores an (s_0, s_h) pair at ITS OWN horizon.  s_0 is the
        # conditioning frame.  A-del's horizon is s_del = the LAST generated
        # frame (WM clips span [0, s_del]).  A-std was trained on (s_0, s_std)
        # with s_std = end of the push, arm mid-motion -- scoring it on the
        # settled last frame would evaluate the "standard metric" on input it
        # never saw, and every C3 number would be an artefact.  The clip's
        # frame_indices (video-rate) locate the frame nearest s_std per item.
        scores = score_generated(name, g, args, dev)
        absent = load_real_clips(g["tuple_index"], args.geometry, "ABSENT", args.clips)
        og = object_region_error(g["generated"], g["real"], absent)
        if not np.isfinite(og).all():
            raise RuntimeError(f"{name}: empty object mask for {int((~np.isfinite(og)).sum())} items")
        scores["object_gt"] = og
        print(f"  {name:24s} object-region GT={og.mean():.5f}", flush=True)
        for k, v in scores.items():
            arrays[f"{name}_{k}"] = v
        arrays[f"{name}_tuple_index"] = g["tuple_index"]
        per_model[name] = scores

    results: dict = {"prereg_lock": lock.to_dict(), "geometry": args.geometry}

    # ---- ladder verification (do this FIRST) -----------------------------
    present = [n for n in names if n in per_model]
    # Every ladder model must be scored on the SAME held-out tuples (§9-H), or
    # the per-model means (and the paired effects) compare different items.
    tuple_sets = {n: set(arrays[f"{n}_tuple_index"].tolist()) for n in present}
    if len({frozenset(s) for s in tuple_sets.values()}) > 1:
        sizes = {n: len(s) for n, s in tuple_sets.items()}
        raise RuntimeError(f"ladder models were scored on different tuple sets: {sizes}; "
                           f"finish/redo generation so every model has the same items")
    if len(present) >= 3:
        gt_key = "object_gt" if gate_gt == "object_region" else "ground_truth"
        gts = [float(per_model[n][gt_key].mean()) for n in present]
        rho = spearman(np.arange(len(present)), gts)
        results["ladder_monotone"] = {
            "models": present, "ground_truth_means": gts, "spearman": rho, "gt_metric": gt_key,
            "object_gt_means": [float(per_model[n]["object_gt"].mean()) for n in present],
            "whole_clip_gt_means": [float(per_model[n]["ground_truth"].mean()) for n in present],
            "passed": bool(rho > 0.9),
            "note": "if this fails the ladder is not a ladder and the between-model "
                    "C3(b) claims are void (§9-H)",
        }
        print(f"\n  ladder monotonicity: spearman={rho:+.3f}  "
              f"{'PASS' if rho > 0.9 else 'FAIL -- between-model claims are VOID'}")

    # ---- C3(a) within-model ---------------------------------------------
    results["within_model"] = {}
    for n in present:
        r = spearman(per_model[n]["standard"], per_model[n]["ogaf"])
        results["within_model"][n] = {"spearman": r, "disagreement": 1.0 - r}
        print(f"  C3(a) {n:24s} Spearman(standard, OG-AF) = {r:+.3f}")

    # ---- C3(b) between-model --------------------------------------------
    results["between_model"] = {}
    ladder_ok = bool(results.get("ladder_monotone", {}).get("passed", False))
    if not ladder_ok:
        # §9-H: if the ladder is not a ladder, between-model claims are void
        # and are NOT reported -- not computed-and-caveated.
        results["between_model"]["void"] = ("ladder not verified monotone; C3(b) is "
                                            "void and was not computed (§9-H)")
        print("\n  C3(b): VOID -- ladder not verified monotone; not computed (§9-H)")
    if ladder_ok and c3b["base"] in per_model and c3b["corrupted"] in per_model:
        a, b = per_model[c3b["base"]], per_model[c3b["corrupted"]]
        if ladder.get("addendum"):
            pc = paired_contrast(a["standard"], b["standard"], a["ogaf"], b["ogaf"],
                                 arrays[f"{c3b['base']}_tuple_index"],
                                 arrays[f"{c3b['corrupted']}_tuple_index"])
            sup = pc["d_ogaf_ci"][0] > 0 and pc["delta_ci"][0] > 0
            ref = pc["delta_ci"][1] < 0
            pc["verdict"] = "SUPPORTED" if sup else ("REFUTED" if ref else "INCONCLUSIVE")
            results["c3b_h2"] = pc
            print(f"\n  C3(b)-H2 (prereg_addendum_H2.md §5): d_std={pc['d_std']:+.3f} "
                  f"{pc['d_std_ci']}  d_ogaf={pc['d_ogaf']:+.3f} {pc['d_ogaf_ci']}  "
                  f"delta={pc['delta']:+.3f} {pc['delta_ci']}  -> {pc['verdict']}")
        results["between_model"]["corrupted_vs_base"] = {
            "standard_effect": cohens_d(b["standard"], a["standard"]),
            "standard_clean_effect": cohens_d(b["standard_clean"], a["standard_clean"]),
            "ogaf_effect": cohens_d(b["ogaf"], a["ogaf"]),
            "dino_effect": cohens_d(b["dino"], a["dino"]),
            "ground_truth_effect": cohens_d(b["ground_truth"], a["ground_truth"]),
        }
        # Paired companion to each unpaired d: same tuples, per-tuple differences.
        ta, tb = arrays[f"{c3b['base']}_tuple_index"], arrays[f"{c3b['corrupted']}_tuple_index"]
        results["between_model"]["corrupted_vs_base_paired"] = {
            k: paired_effect(a[k], ta, b[k], tb)
            for k in ("standard", "standard_clean", "ogaf", "dino", "ground_truth")
        }
        e = results["between_model"]["corrupted_vs_base"]
        ep = results["between_model"]["corrupted_vs_base_paired"]
        print(f"\n  C3(b) physics-corrupted vs base  (unpaired d = registered; paired d on "
              f"the same {ep['ogaf'].get('n_pairs', 0)} tuples alongside):")
        for k, lab in (("standard", "standard-metric"), ("ogaf", "OG-AF"), ("dino", "DINO @ s_del"),
                       ("ground_truth", "ground truth")):
            p = ep[k]
            if "d_paired" in p:
                print(f"    {lab:16s} paired d = {p['d_paired']:+.3f} "
                      f"[{p['ci_low']:+.3f}, {p['ci_high']:+.3f}]")
        print(f"\n  C3(b) physics-corrupted vs base:")
        print(f"    standard-metric  d = {e['standard_effect']:+.3f}   "
              f"(prediction: near zero -- the metric cannot see wrong physics)")
        print(f"    OG-AF            d = {e['ogaf_effect']:+.3f}   "
              f"(prediction: clearly separated)")
        print(f"    DINO @ s_del     d = {e['dino_effect']:+.3f}")
        print(f"    ground truth     d = {e['ground_truth_effect']:+.3f}   "
              f"(privileged; confirms the models really do differ)")

    # ---- ladder.yaml controls (§9-H) ------------------------------------
    # appearance_gap: free-space (ABSENT) rollouts have no object, so contact
    # physics cannot diverge by construction; whatever the IDMs and the DINO
    # distance read there is domain gap (sim -> generated rendering), not
    # physics.  domain_shift: the same feature-space distance on the INTERACT
    # generations, next to the clean-vs-augmented IDM delta.  Generated with
    # `wm_generate.py --condition ABSENT --n 300` per model (HANDOFF §2).
    results["domain_gap"] = {}
    results["domain_shift"] = {}
    for name in present:
        ds_ = per_model[name]
        results["domain_shift"][name] = {
            "dino_generated_vs_real": float(ds_["dino"].mean()),
            "standard_augmented": float(ds_["standard"].mean()),
            "standard_clean": float(ds_["standard_clean"].mean()),
            "clean_minus_augmented": float(ds_["standard_clean"].mean() - ds_["standard"].mean()),
        }
        try:
            ga = load_generated(name, generated_subdir(args.geometry, "ABSENT"),
                                args.generated_root)
        except FileNotFoundError:
            results["domain_gap"][name] = {"missing": "no ABSENT (free-space) generations"}
            continue
        sa = score_generated(f"{name}/ABSENT", ga, args, dev)
        for k, v in sa.items():
            arrays[f"{name}_ABSENT_{k}"] = v
        arrays[f"{name}_ABSENT_tuple_index"] = ga["tuple_index"]
        results["domain_gap"][name] = {
            "n": int(len(ga["tuple_index"])),
            **{k: float(v.mean()) for k, v in sa.items()},
            "note": "no object -> physics divergence is zero by construction; these are "
                    "the rendering/domain-gap floors of each metric",
        }
        print(f"  control {name:24s} ABSENT: standard={sa['standard'].mean():.5f}  "
              f"ogaf={sa['ogaf'].mean():.5f}  dino={sa['dino'].mean():.5f}  "
              f"gt={sa['ground_truth'].mean():.5f}")

    args.out.mkdir(parents=True, exist_ok=True)
    provenance.save_run(args.out, EXPERIMENT, arrays,
                        extra={"results": results, "encoder": args.encoder,
                               "idm_seeds": args.seeds},
                        arrays_name="ladder.npz")
    with open(args.out / "results.json", "w") as fh:
        json.dump(provenance._jsonable(results), fh, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}.  Verdicts: analyze.py --exp h")
    print("\nKnown limitation to state, not hide (§9-H): a single 3-DOF action per\n"
          "rollout is simpler than the per-step action sequences of the critiqued\n"
          "protocols.  Address head-on in limitations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
