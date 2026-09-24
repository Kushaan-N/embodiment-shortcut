"""Generate rollouts from a trained world model, with the §13.5 validators inline.

The expensive failure on a cluster is not a crash -- it is a job that exits 0
and writes unusable output.  Every item is therefore validated on content, never
on exit code, and every item's content checksum goes into ``manifest.json`` so a
corrupted volume or an interrupted rsync becomes detectable later.

Idempotence: one atomically-written file per item (tmp -> fsync -> rename),
deterministic ids, skip-if-exists.

    python wm/wm_generate.py --model WM-base --n 1500
    python wm/wm_generate.py --model WM-base --s1        # S1 ladder stage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C  # noqa: E402
import datasets as ds  # noqa: E402
import provenance  # noqa: E402
import scene  # noqa: E402
from wm.vae import CompactVAE  # noqa: E402
from wm.wm_train import VideoWorldModel  # noqa: E402

LADDER_PATH = Path(__file__).resolve().parent / "ladder.yaml"


def resolve_checkpoint(model_name: str, ckpt_root: Path) -> Path:
    """Map a ladder model name to the checkpoint that DEFINES it (wm/ladder.yaml).

    WM-base-100 / -30 / -10 are checkpoints of ONE training run -- trained
    under the base name (``ladder.base.name``, what wm_train.sbatch submits)
    -- at the ladder's ``checkpoint_at`` steps.  The override models are
    their own runs and use ``final.pt``.  Nothing else maps these names, so
    without this every ``--model WM-base-100`` died in torch.load.
    """
    ckpt_root = Path(ckpt_root)
    ladder = yaml.safe_load(LADDER_PATH.read_text())
    base_name = ladder["base"]["name"]
    for m in ladder["models"]:
        if m["name"] != model_name:
            continue
        if "checkpoint_at" in m:
            step = int(m["checkpoint_at"])
            p = ckpt_root / base_name / f"ckpt_step{step:07d}.pt"
            if not p.exists():
                raise FileNotFoundError(
                    f"{model_name} is {base_name} at step {step}: {p} missing "
                    f"(train {base_name} past step {step}; checkpoints are kept)")
            return p
        return ckpt_root / model_name / "final.pt"
    return ckpt_root / model_name / "final.pt"     # e.g. the base run itself (S1)


def load_model(ckpt_path: Path, device, vae_path: Path | None = None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vpath = vae_path or ck.get("extra", {}).get("vae") or (
        C.CHECKPOINT_ROOT / "wm" / "vae" / "vae.pt")
    vck = torch.load(vpath, map_location="cpu", weights_only=False)
    vae = CompactVAE(vck["latent_channels"]).to(device).eval()
    vae.load_state_dict(vck["state_dict"])
    t = ck["cfg"]["transformer"]
    grid = C.RENDER_SIZE // ck["cfg"]["vae"]["downsample"]
    model = VideoWorldModel(
        latent_channels=vck["latent_channels"], grid=grid, patch=t["patch"],
        dim=t["dim"], depth=t["depth"], heads=t["heads"],
        max_frames=t["max_frames"],
    ).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model, vae, ck


@torch.no_grad()
def generate_video(model, vae, first_frame: np.ndarray, action: np.ndarray,
                   n_frames: int, device) -> np.ndarray:
    """Autoregressive rollout: one conditioning frame -> (n_frames, H, W, 3) uint8.

    Genuinely autoregressive -- each predicted latent is fed back in and the
    next is predicted from the model's OWN output, never from ground truth.
    Teacher-forcing here would make a broken model look good and is exactly the
    failure the S1 two-action divergence check cannot catch on its own.
    """
    x = torch.from_numpy(np.ascontiguousarray(first_frame))[None]
    x = x.permute(0, 3, 1, 2).contiguous().float().to(device) / 255.0
    mu, _ = vae.encode(x)                                  # (1, Cl, G, G)
    seq = mu[None]                                         # (1, 1, Cl, G, G)
    a = torch.from_numpy(np.asarray(action, dtype=np.float32))[None].to(device)

    for _ in range(n_frames - 1):
        pred = model(seq, a)                               # (1, t, Cl, G, G)
        seq = torch.cat([seq, pred[:, -1:]], dim=1)        # append the newest frame

    lat = seq[0]                                           # (n_frames, Cl, G, G)
    imgs = vae.decode(lat).permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()
    return (imgs * 255).round().astype(np.uint8)


# ==========================================================================
# §13.5 validators -- content, never exit codes
# ==========================================================================


def validate_item(frames: np.ndarray, conditioning: np.ndarray, expected_count: int,
                  *, pixel_std_floor: float = 2.0, diff_floor: float = 1.0) -> dict:
    checks = {}

    def add(name, passed, value, threshold):
        checks[name] = {"passed": bool(passed), "value": float(value),
                        "threshold": float(threshold)}

    add("frame_count_exact", len(frames) == expected_count, len(frames), expected_count)
    stds = [float(f.astype(np.float64).std()) for f in frames]
    add("per_frame_pixel_std_above_floor", min(stds) > pixel_std_floor,
        min(stds), pixel_std_floor)
    if len(frames) > 1:
        diffs = [float(np.abs(frames[i + 1].astype(np.int32) - frames[i].astype(np.int32)).mean())
                 for i in range(len(frames) - 1)]
        add("consecutive_frame_difference_above_floor", min(diffs) > diff_floor,
            min(diffs), diff_floor)
    d0 = float(np.abs(frames[0].astype(np.int32) - conditioning.astype(np.int32)).mean())
    add("frame0_close_to_conditioning", d0 < 40.0, d0, 40.0)
    checks["_all_passed"] = all(v["passed"] for v in checks.values() if isinstance(v, dict))
    return checks


def s1_checks(model, vae, first_frame, action, n_frames, device) -> dict:
    """Determinism and two-action divergence -- run at S1, not S2 (§13.3).

    Action conditioning being silently unwired is the one bug that invalidates
    every downstream Experiment H number, and it is detectable with two
    generations.  Catching it at S1 costs one salloc; catching it at S4 costs
    the whole allocation.
    """
    a = generate_video(model, vae, first_frame, action, n_frames, device)
    b = generate_video(model, vae, first_frame, action, n_frames, device)
    determinism = float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())

    lo, hi = C.action_ranges_array()
    other = lo + (hi - lo) * 0.9 if float(action[0]) < 0.5 * (lo[0] + hi[0]) else lo
    c = generate_video(model, vae, first_frame, np.asarray(other, dtype=np.float32),
                       n_frames, device)
    divergence = float(np.abs(a.astype(np.int32) - c.astype(np.int32)).mean())

    # A video model that ignores time would emit a still: every frame equal to
    # the conditioning frame.  That passes determinism AND could pass action
    # divergence, so it needs its own check.
    motion = float(np.abs(a[1:].astype(np.int32) - a[:-1].astype(np.int32)).mean())

    return {
        "determinism": {"passed": determinism < 1e-6, "value": determinism,
                        "threshold": 1e-6,
                        "note": "same seed twice must give identical output"},
        "two_action_divergence": {
            "passed": divergence > 2.0, "value": divergence, "threshold": 2.0,
            "note": "two different actions must give materially different video; "
                    "failing this means action conditioning is unwired and every "
                    "downstream H number is void (§13.3)"},
        "temporal_motion": {
            "passed": motion > 0.5, "value": motion, "threshold": 0.5,
            "note": "generated video must actually move; a model emitting a still "
                    "would pass both checks above"},
    }


# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="WM-base")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--s1", action="store_true", help="S1 ladder stage: 1 item + S1 checks")
    ap.add_argument("--out", type=Path, default=C.DATA_ROOT / "generated")
    ap.add_argument("--geometry", default="box", choices=C.GEOMETRIES)
    ap.add_argument("--clips", type=Path, default=C.DATA_ROOT / "wm_clips")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or resolve_checkpoint(args.model, C.CHECKPOINT_ROOT / "wm")
    print(f"  {args.model} <- {ckpt}", flush=True)
    model, vae, ck = load_model(ckpt, dev)
    n_frames = int(ck["cfg"]["transformer"]["max_frames"])

    # The SAME held-out action set for every model on the ladder (§9-H), and the
    # same REAL video to score against -- the WM clips, not the horizon frames.
    clip_root = Path(args.clips) / args.geometry / "INTERACT"
    items = sorted(clip_root.glob("*.npz"))
    if not items:
        raise SystemExit(f"no WM clips under {clip_root}; run wm/render_clips.py")
    held_out = []
    for f in items:
        with np.load(f) as z:
            if str(z["split"]) == "test":
                held_out.append((f, z["action"].copy(), int(z["tuple_index"])))
    n = min(args.n, len(held_out))
    if n < args.n:
        # The test split is 10% of the corpus, so --n 1500 is not reachable;
        # every model is scored on the same n either way, but say so.
        print(f"  WARNING: only {n} held-out test clips exist (requested --n {args.n}); "
              f"every ladder model is scored on these same {n}", flush=True)
    idx = np.arange(n)[args.shard :: args.n_shards]

    outdir = Path(args.out) / args.model / args.geometry
    outdir.mkdir(parents=True, exist_ok=True)

    if args.s1:
        with np.load(held_out[0][0]) as z:
            first = z["clip"][0].copy()
        res = s1_checks(model, vae, first, held_out[0][1], n_frames, dev)
        print(json.dumps(res, indent=2))
        ok = all(v["passed"] for v in res.values())
        (outdir / "s1_checks.json").write_text(json.dumps(res, indent=2))
        print(f"S1: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 2

    manifest, n_fail, consecutive_fail = [], 0, 0
    for k, i in enumerate(idx):
        clip_path, action, tup = held_out[i]
        item_id = f"{tup:07d}"
        path = outdir / f"{item_id}.npz"
        if path.exists():
            continue
        with np.load(clip_path) as z:
            real = z["clip"].copy()
            frame_idx = z["frame_indices"].astype(np.int64).copy()
            s_del_step = np.int64(z["s_del_step"])
        gen = generate_video(model, vae, real[0], action, n_frames, dev)
        checks = validate_item(gen, real[0], expected_count=n_frames)
        digest = hashlib.sha256(gen.tobytes()).hexdigest()
        # frame_indices travel with the item: exp_h must score the STANDARD
        # metric at the frame nearest s_std, not at the settled last frame.
        provenance.save_arrays(path, generated=gen, conditioning=real[:1],
                               action=action, real_target=real,
                               frame_indices=frame_idx, s_del_step=s_del_step)
        manifest.append({"id": item_id, "sha256": digest,
                         "passed": checks["_all_passed"], "checks": checks})
        if not checks["_all_passed"]:
            n_fail += 1
            consecutive_fail += 1
            if k < 5 and consecutive_fail == 5:
                raise SystemExit("first 5 items all failed validation -- aborting task "
                                 "rather than filling scratch with unusable output (§13.5)")
        else:
            consecutive_fail = 0
        if k % 100 == 0:
            print(f"  {k}/{len(idx)}  failures={n_fail}", flush=True)

    mpath = outdir / f"manifest_shard{args.shard:03d}.json"
    rate = n_fail / max(len(manifest), 1)
    mpath.write_text(json.dumps({"model": args.model, "geometry": args.geometry,
                                 "n": len(manifest), "n_failed": n_fail,
                                 "failure_rate": rate, "items": manifest}, indent=1))
    print(f"wrote {mpath}: {len(manifest)} items, {n_fail} failed ({rate:.1%})")
    if rate > 0.05:
        print("failure rate above 5% -- diagnose before analysing (§13.5)")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
