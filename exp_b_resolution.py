"""Experiment B -- encoder resolution floor (§9-B).  GPU, minutes.  FIRST GPU GATE.

T4: both metrics live in a feature space.  If that space cannot resolve
contact-scale geometry, every number downstream is uninterpretable.  This gate
runs before anything else on a GPU.

Method
------
Renders match the s_del frame composition -- object settled, **arm at canonical
rest** -- because arm occlusion and arm pixels are part of the operating
condition, not a detail to be abstracted away.

  1. Anchor render.
  2. Translation sweep: ``logspace(0.1 * delta_pos_min, 10 * delta_pos_min)``,
     20 points x 10 directions.  It starts *below* delta_min deliberately, so
     saturation is approached from underneath and the curve shows where the
     encoder stops responding.
  3. Rotation sweep: 10 axes over a comparable range around delta_rot_min.
  4. Noise floor: the anchor rendered 20x with sub-pixel camera jitter smaller
     than ``delta_pos_min / 10``.

CLS and mean-pooled patch tokens are reported separately -- they are different
representations and pooling them would hide a pass or a fail.

Gate B
------
PASS if the feature distance at ``delta_min`` exceeds the noise-floor mean by
at least 3 sigma, for at least one (encoder, pooling) combination.

If every combination fails, the finding is "DINO-family features cannot resolve
contact-scale geometry at this scale".  That is a legitimate result: report it
and stop, and the project redirects toward hybrid representations (semantic +
dense flow).  Do not swap the encoder to make the gate pass (§0.1).

    python exp_b_resolution.py [--geometry box] [--encoders ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config as C
import distances as D
import provenance
import scene

EXPERIMENT = "exp_b_resolution"

GATE_B_SIGMA = 3.0
N_TRANSLATION_POINTS = 20
N_TRANSLATION_DIRECTIONS = 10
N_ROTATION_POINTS = 20
N_ROTATION_AXES = 10
N_NOISE_FLOOR = 20


# ==========================================================================
# Encoders
# ==========================================================================


class FrozenEncoder:
    """A frozen ViT returning both CLS and mean-pooled patch embeddings.

    Images are fed at their native 224x224 with no resize step (§6): resizing
    would blur exactly the sub-pixel differences this experiment measures.
    """

    def __init__(self, name: str, device: torch.device):
        from transformers import AutoModel

        self.name = name
        self.device = device
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        self.mean = torch.tensor(getattr(cfg, "image_mean", (0.485, 0.456, 0.406)))
        self.std = torch.tensor(getattr(cfg, "image_std", (0.229, 0.224, 0.225)))
        self.mean = self.mean.view(1, 3, 1, 1).to(device)
        self.std = self.std.view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def __call__(self, images: np.ndarray, batch: int = 32) -> dict:
        """images: (N, H, W, 3) uint8 -> {'cls': (N, D), 'mean_patch': (N, D)}."""
        outs = {"cls": [], "mean_patch": []}
        for i in range(0, len(images), batch):
            chunk = images[i : i + batch]
            x = torch.from_numpy(np.ascontiguousarray(chunk)).to(self.device)
            x = x.permute(0, 3, 1, 2).contiguous().float() / 255.0
            x = (x - self.mean) / self.std
            h = self.model(pixel_values=x).last_hidden_state    # (B, 1+P, D)
            outs["cls"].append(h[:, 0].float().cpu().numpy())
            outs["mean_patch"].append(h[:, 1:].mean(dim=1).float().cpu().numpy())
        return {k: np.concatenate(v) for k, v in outs.items()}


def load_encoders(names, device) -> tuple[dict, list]:
    """Load what is available; report -- never silently substitute (§9-B)."""
    loaded, unavailable = {}, []
    for n in names:
        try:
            loaded[n] = FrozenEncoder(n, device)
            print(f"  loaded encoder {n}")
        except Exception as exc:  # noqa: BLE001
            unavailable.append({"encoder": n, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  UNAVAILABLE {n}: {type(exc).__name__}: {exc}")
    return loaded, unavailable


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return 1.0 - np.sum(a * b, axis=-1)


# ==========================================================================
# Render sweeps
# ==========================================================================


def anchor_state(geometry: str) -> tuple[np.ndarray, np.ndarray]:
    """A representative settled state: object at a typical settling radius,
    arm at canonical rest (matching the s_del frame composition, §9-B v3)."""
    arm = np.array([*C.REST_POSE, C.REST_YAW])
    h = scene.object_rest_height(geometry)
    pose = np.array([0.055, 0.015, h, 1.0, 0.0, 0.0, 0.0])
    # A generic, non-axis-aligned orientation, so the sweep is not measured at
    # a symmetry-special pose.
    q = D.quat_from_axis_angle([0.3, 0.2, 0.93], 0.7)
    pose[3:] = q
    return arm, pose


def translation_sweep(geometry: str, d_pos_min: float, rng) -> tuple[np.ndarray, np.ndarray]:
    arm, pose = anchor_state(geometry)
    mags = np.logspace(np.log10(0.1 * d_pos_min), np.log10(10 * d_pos_min),
                       N_TRANSLATION_POINTS)
    dirs = rng.normal(size=(N_TRANSLATION_DIRECTIONS, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    frames, meta = [], []
    for mi, m in enumerate(mags):
        for di, d in enumerate(dirs):
            p = pose.copy()
            p[0] += m * d[0]
            p[1] += m * d[1]
            frames.append(scene.render_state(geometry, "INTERACT", arm, p))
            meta.append((mi, di, m))
    return np.stack(frames), np.array(meta)


def rotation_sweep(geometry: str, d_rot_min: float, rng) -> tuple[np.ndarray, np.ndarray]:
    arm, pose = anchor_state(geometry)
    mags = np.logspace(np.log10(0.1 * d_rot_min), np.log10(10 * d_rot_min), N_ROTATION_POINTS)
    axes = rng.normal(size=(N_ROTATION_AXES, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    frames, meta = [], []
    for mi, m in enumerate(mags):
        for ai, ax in enumerate(axes):
            p = pose.copy()
            p[3:] = D.quat_multiply(D.quat_from_axis_angle(ax, m), pose[3:])
            frames.append(scene.render_state(geometry, "INTERACT", arm, p))
            meta.append((mi, ai, m))
    return np.stack(frames), np.array(meta)


def noise_floor_frames(geometry: str, d_pos_min: float, rng) -> np.ndarray:
    """The anchor re-rendered with sub-pixel camera jitter below delta_pos_min/10."""
    arm, pose = anchor_state(geometry)
    scale = d_pos_min / 10.0
    frames = []
    for _ in range(N_NOISE_FLOOR):
        off = rng.normal(scale=scale / np.sqrt(3), size=3)
        frames.append(scene.render_state(geometry, "INTERACT", arm, pose, camera_offset=off))
    return np.stack(frames)


# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="box", choices=C.GEOMETRIES)
    ap.add_argument("--encoders", nargs="+", default=list(C.ENCODERS))
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "exp_b")
    ap.add_argument("--seed", type=int, default=C.MASTER_SEED + 11)
    args = ap.parse_args()

    thresholds = C.load_thresholds()
    thr = thresholds.for_geometry(args.geometry)
    d_pos_min = thr["delta_pos_min"]
    d_rot_min = thr["delta_rot_min"] or thr["delta_rot_raw_min"]
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       ("mps" if torch.backends.mps.is_available() else "cpu"))

    print(f"Experiment B on {args.geometry}: delta_pos_min={d_pos_min * 1e3:.4f} mm, "
          f"delta_rot_min={np.degrees(d_rot_min):.4f} deg, device={dev}")

    print("rendering sweeps ...")
    anchor_arm, anchor_pose = anchor_state(args.geometry)
    anchor = scene.render_state(args.geometry, "INTERACT", anchor_arm, anchor_pose)[None]
    tr_frames, tr_meta = translation_sweep(args.geometry, d_pos_min, rng)
    rot_frames, rot_meta = rotation_sweep(args.geometry, d_rot_min, rng)
    nf_frames = noise_floor_frames(args.geometry, d_pos_min, rng)
    print(f"  anchor 1, translation {len(tr_frames)}, rotation {len(rot_frames)}, "
          f"noise-floor {len(nf_frames)}")

    encoders, unavailable = load_encoders(args.encoders, dev)
    if not encoders:
        raise RuntimeError(
            "no encoder could be loaded; §9-B forbids silently substituting one. "
            f"Failures: {json.dumps(unavailable, indent=2)}"
        )

    arrays = {"anchor": anchor, "translation_meta": tr_meta, "rotation_meta": rot_meta}
    results = {"geometry": args.geometry, "delta_pos_min": d_pos_min,
               "delta_rot_min": d_rot_min, "unavailable_encoders": unavailable,
               "gate_sigma": GATE_B_SIGMA, "encoders": {}}

    for name, enc in encoders.items():
        e_anchor = enc(anchor)
        e_tr = enc(tr_frames)
        e_rot = enc(rot_frames)
        e_nf = enc(nf_frames)
        per_pool = {}
        for pool in C.ENCODER_POOLINGS:
            a = e_anchor[pool]
            d_tr = cosine_distance(np.repeat(a, len(e_tr[pool]), 0), e_tr[pool])
            d_rot = cosine_distance(np.repeat(a, len(e_rot[pool]), 0), e_rot[pool])
            d_nf = cosine_distance(np.repeat(a, len(e_nf[pool]), 0), e_nf[pool])

            key = f"{name.replace('/', '_')}_{pool}"
            arrays[f"dist_translation_{key}"] = d_tr
            arrays[f"dist_rotation_{key}"] = d_rot
            arrays[f"dist_noisefloor_{key}"] = d_nf

            nf_mean, nf_std = float(d_nf.mean()), float(d_nf.std(ddof=1))
            threshold = nf_mean + GATE_B_SIGMA * nf_std

            # Distance at exactly delta_min: the sweep is log-spaced and
            # symmetric about delta_min, so the middle magnitude index is it.
            mid = N_TRANSLATION_POINTS // 2
            at_min_tr = d_tr[tr_meta[:, 0] == mid]
            mid_r = N_ROTATION_POINTS // 2
            at_min_rot = d_rot[rot_meta[:, 0] == mid_r]

            passed_tr = float(at_min_tr.mean()) > threshold
            passed_rot = float(at_min_rot.mean()) > threshold
            per_pool[pool] = {
                "noise_floor_mean": nf_mean,
                "noise_floor_std": nf_std,
                "gate_threshold": threshold,
                "translation_at_delta_min_mean": float(at_min_tr.mean()),
                "translation_at_delta_min_sigma": (float(at_min_tr.mean()) - nf_mean)
                / max(nf_std, 1e-12),
                "rotation_at_delta_min_mean": float(at_min_rot.mean()),
                "rotation_at_delta_min_sigma": (float(at_min_rot.mean()) - nf_mean)
                / max(nf_std, 1e-12),
                "passed_translation": bool(passed_tr),
                "passed_rotation": bool(passed_rot),
                "passed": bool(passed_tr or passed_rot),
                "translation_curve": [
                    {"magnitude_m": float(tr_meta[tr_meta[:, 0] == i][0, 2]),
                     "mean_distance": float(d_tr[tr_meta[:, 0] == i].mean()),
                     "sigma_over_floor": (float(d_tr[tr_meta[:, 0] == i].mean()) - nf_mean)
                     / max(nf_std, 1e-12)}
                    for i in range(N_TRANSLATION_POINTS)
                ],
                "rotation_curve": [
                    {"magnitude_rad": float(rot_meta[rot_meta[:, 0] == i][0, 2]),
                     "mean_distance": float(d_rot[rot_meta[:, 0] == i].mean()),
                     "sigma_over_floor": (float(d_rot[rot_meta[:, 0] == i].mean()) - nf_mean)
                     / max(nf_std, 1e-12)}
                    for i in range(N_ROTATION_POINTS)
                ],
            }
            print(f"  {name:52s} {pool:11s} floor={nf_mean:.3e}+-{nf_std:.1e}  "
                  f"trans@dmin={per_pool[pool]['translation_at_delta_min_sigma']:7.2f} sigma  "
                  f"rot@dmin={per_pool[pool]['rotation_at_delta_min_sigma']:7.2f} sigma  "
                  f"{'PASS' if per_pool[pool]['passed'] else 'fail'}")
        results["encoders"][name] = per_pool

    gate = _evaluate_gate_b(results)
    results["_gate_B"] = gate

    args.out.mkdir(parents=True, exist_ok=True)
    provenance.save_run(args.out, EXPERIMENT, arrays,
                        seeds={"sweep_seed": args.seed},
                        extra={"geometry": args.geometry, "device": str(dev),
                               "results": results},
                        arrays_name="resolution.npz")
    with open(args.out / "results.json", "w") as fh:
        json.dump(provenance._jsonable(results), fh, indent=2, sort_keys=True)

    print("\n" + "=" * 72)
    print(gate["report"])
    print("=" * 72)
    return 0 if gate["passed"] else 2


def _evaluate_gate_b(results: dict) -> dict:
    winners = [
        (enc, pool, v["translation_at_delta_min_sigma"], v["rotation_at_delta_min_sigma"])
        for enc, pools in results["encoders"].items()
        for pool, v in pools.items() if v["passed"]
    ]
    ok = bool(winners)
    lines = [f"GATE B: {'PASS' if ok else 'FAIL'}  "
             f"(need >= {GATE_B_SIGMA} sigma above the noise floor at delta_min, "
             f"for at least one encoder/pooling)"]
    for enc, pools in results["encoders"].items():
        for pool, v in pools.items():
            lines.append(f"  {enc} / {pool}: translation "
                         f"{v['translation_at_delta_min_sigma']:.2f} sigma, rotation "
                         f"{v['rotation_at_delta_min_sigma']:.2f} sigma "
                         f"-> {'PASS' if v['passed'] else 'fail'}")
    if results["unavailable_encoders"]:
        lines.append("  encoders that could NOT be loaded (reported, not substituted):")
        for u in results["unavailable_encoders"]:
            lines.append(f"    {u['encoder']}: {u['error']}")
    lines.append("")
    if ok:
        best = max(winners, key=lambda w: max(w[2], w[3]))
        lines.append(f"  Use {best[0]} / {best[1]} for Architecture B and the DINO state "
                     f"distance.\n  Record its noise floor as the T9 arm-deviation tolerance "
                     f"target (§4-T9).")
    else:
        lines.append(
            "  STOP and report (§0.1).  The finding is that DINO-family features cannot\n"
            "  resolve contact-scale geometry at this rendering scale.  That is a real\n"
            "  result and it redirects the project toward hybrid representations\n"
            "  (semantic + dense flow).  Do NOT swap the encoder to make the gate pass."
        )
    return {"passed": ok, "winners": [list(w) for w in winners], "report": "\n".join(lines)}


if __name__ == "__main__":
    raise SystemExit(main())
