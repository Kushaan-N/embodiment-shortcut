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


def load_generated(model_name: str, geometry: str, root: Path) -> dict:
    d = Path(root) / model_name / geometry
    items = sorted(d.glob("*.npz"))
    if not items:
        raise FileNotFoundError(f"no generated rollouts under {d}")
    gen, cond, act, real, ids = [], [], [], [], []
    for p in items:
        with np.load(p) as z:
            gen.append(z["generated"])
            cond.append(z["conditioning"])
            act.append(z["action"])
            real.append(z["real_target"])
        ids.append(int(p.stem))
    return {"generated": np.stack(gen), "conditioning": np.stack(cond),
            "action": np.stack(act), "real": np.stack(real),
            "tuple_index": np.asarray(ids)}


@torch.no_grad()
def score_with_idm(variant: str, seed: int, batch: np.ndarray, actions: np.ndarray,
                   *, augmented: bool, ckpt_root: Path, device) -> np.ndarray:
    """Per-rollout normalised MAE from a trained IDM on (s_0, generated) pairs."""
    tag = f"{variant}__seed{seed}__full" + ("__aug" if augmented else "")
    ck = torch.load(Path(ckpt_root) / tag / "checkpoint.pt", map_location="cpu",
                    weights_only=False)
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


def ground_truth_error(generated: np.ndarray, real: np.ndarray) -> np.ndarray:
    """Privileged comparison against the simulator's own settled frame."""
    g = generated.astype(np.float64)
    r = real.astype(np.float64)
    return np.abs(g - r).mean(axis=(1, 2, 3)) / 255.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generated-root", type=Path, default=C.DATA_ROOT / "generated")
    ap.add_argument("--ckpt-root", type=Path, default=C.CHECKPOINT_ROOT)
    ap.add_argument("--geometry", default="box", choices=C.GEOMETRIES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--encoder", default=C.ENCODERS[0])
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "exp_h")
    args = ap.parse_args()

    lock = prereg_lock.require("h", [args.out / "results.json"])
    print(lock.render(), "\n")

    ladder = yaml.safe_load(LADDER_PATH.read_text())
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
        pairs = np.stack([g["conditioning"][:, 0], g["generated"]], axis=1)

        std_e = np.mean([score_with_idm("A-std", s, pairs, g["action"], augmented=True,
                                        ckpt_root=args.ckpt_root, device=dev)
                         for s in args.seeds], axis=0)
        del_e = np.mean([score_with_idm("A-del", s, pairs, g["action"], augmented=True,
                                        ckpt_root=args.ckpt_root, device=dev)
                         for s in args.seeds], axis=0)
        std_clean = np.mean([score_with_idm("A-std", s, pairs, g["action"], augmented=False,
                                            ckpt_root=args.ckpt_root, device=dev)
                             for s in args.seeds], axis=0)
        dino = dino_state_distance(g["generated"], g["real"], args.encoder, dev)
        gt = ground_truth_error(g["generated"], g["real"])

        for k, v in [("standard", std_e), ("ogaf", del_e), ("standard_clean", std_clean),
                     ("dino", dino), ("ground_truth", gt),
                     ("tuple_index", g["tuple_index"])]:
            arrays[f"{name}_{k}"] = v
        per_model[name] = {"standard": std_e, "ogaf": del_e, "standard_clean": std_clean,
                           "dino": dino, "ground_truth": gt}
        print(f"  {name:24s} standard={std_e.mean():.5f}  ogaf={del_e.mean():.5f}  "
              f"dino={dino.mean():.5f}  gt={gt.mean():.5f}")

    results: dict = {"prereg_lock": lock.to_dict(), "geometry": args.geometry}

    # ---- ladder verification (do this FIRST) -----------------------------
    present = [n for n in names if n in per_model]
    if len(present) >= 3:
        gts = [per_model[n]["ground_truth"].mean() for n in present]
        rho = spearman(np.arange(len(present)), gts)
        results["ladder_monotone"] = {
            "models": present, "ground_truth_means": gts, "spearman": rho,
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
    if "WM-base-100" in per_model and "WM-physics-corrupted" in per_model:
        a, b = per_model["WM-base-100"], per_model["WM-physics-corrupted"]
        results["between_model"]["corrupted_vs_base"] = {
            "standard_effect": cohens_d(b["standard"], a["standard"]),
            "standard_clean_effect": cohens_d(b["standard_clean"], a["standard_clean"]),
            "ogaf_effect": cohens_d(b["ogaf"], a["ogaf"]),
            "dino_effect": cohens_d(b["dino"], a["dino"]),
            "ground_truth_effect": cohens_d(b["ground_truth"], a["ground_truth"]),
        }
        e = results["between_model"]["corrupted_vs_base"]
        print(f"\n  C3(b) physics-corrupted vs base:")
        print(f"    standard-metric  d = {e['standard_effect']:+.3f}   "
              f"(prediction: near zero -- the metric cannot see wrong physics)")
        print(f"    OG-AF            d = {e['ogaf_effect']:+.3f}   "
              f"(prediction: clearly separated)")
        print(f"    DINO @ s_del     d = {e['dino_effect']:+.3f}")
        print(f"    ground truth     d = {e['ground_truth_effect']:+.3f}   "
              f"(privileged; confirms the models really do differ)")

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
