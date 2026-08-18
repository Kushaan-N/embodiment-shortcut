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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C  # noqa: E402
import datasets as ds  # noqa: E402
import provenance  # noqa: E402
import scene  # noqa: E402
from wm.vae import CompactVAE  # noqa: E402
from wm.wm_train import ActionConditionedPredictor  # noqa: E402


def load_model(ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vck = torch.load(ck["vae"], map_location="cpu", weights_only=False)
    vae = CompactVAE(vck["latent_channels"]).to(device).eval()
    vae.load_state_dict(vck["state_dict"])
    t = ck["cfg"]["transformer"]
    grid = C.RENDER_SIZE // vck.get("downsample", 8)
    model = ActionConditionedPredictor(
        latent_channels=vck["latent_channels"], grid=grid, patch=t["patch"],
        dim=t["dim"], depth=t["depth"], heads=t["heads"],
        context_frames=t["context_frames"],
    ).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model, vae, ck


@torch.no_grad()
def generate_one(model, vae, context_frames: np.ndarray, action: np.ndarray, device):
    """context_frames: (ctx, H, W, 3) uint8 -> predicted settled frame (H, W, 3) uint8."""
    x = torch.from_numpy(np.ascontiguousarray(context_frames))
    x = x.permute(0, 3, 1, 2).contiguous().float().to(device) / 255.0
    mu, _ = vae.encode(x)
    ctx = mu[None]                                        # (1, T, Cl, G, G)
    a = torch.from_numpy(np.asarray(action, dtype=np.float32))[None].to(device)
    z = model(ctx, a)
    img = vae.decode(z)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (img * 255).round().astype(np.uint8)


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


def s1_checks(model, vae, context, action, device) -> dict:
    """Determinism and two-action divergence -- run at S1, not S2 (§13.3).

    Action conditioning being silently unwired is the one bug that invalidates
    every downstream Experiment H number, and it is detectable with two
    generations.  Catching it at S1 costs one salloc; catching it at S4 costs
    the whole allocation.
    """
    a = generate_one(model, vae, context, action, device)
    b = generate_one(model, vae, context, action, device)
    determinism = float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())

    lo, hi = C.action_ranges_array()
    other = lo + (hi - lo) * 0.9 if float(action[0]) < 0.5 * (lo[0] + hi[0]) else lo
    c = generate_one(model, vae, context, np.asarray(other, dtype=np.float32), device)
    divergence = float(np.abs(a.astype(np.int32) - c.astype(np.int32)).mean())

    return {
        "determinism": {"passed": determinism < 1e-6, "value": determinism,
                        "threshold": 1e-6,
                        "note": "same seed twice must give identical output"},
        "two_action_divergence": {
            "passed": divergence > 2.0, "value": divergence, "threshold": 2.0,
            "note": "two different actions must give materially different video; "
                    "failing this means action conditioning is unwired and every "
                    "downstream H number is void (§13.3)"},
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
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or (C.CHECKPOINT_ROOT / "wm" / args.model / "final.pt")
    model, vae, ck = load_model(ckpt, dev)
    ctxn = ck["cfg"]["transformer"]["context_frames"]

    # The SAME held-out action set for every model on the ladder (§9-H).
    data = ds.load_split("test", geometry=args.geometry, condition="INTERACT",
                         keys=["frames", "action", "tuple_index"])
    frames, actions, tuples = data["frames"], data["action"], data["tuple_index"]
    n = min(args.n, len(frames))
    idx = np.arange(n)[args.shard :: args.n_shards]

    outdir = Path(args.out) / args.model / args.geometry
    outdir.mkdir(parents=True, exist_ok=True)

    if args.s1:
        res = s1_checks(model, vae, frames[0][:ctxn], actions[0], dev)
        print(json.dumps(res, indent=2))
        ok = all(v["passed"] for v in res.values())
        (outdir / "s1_checks.json").write_text(json.dumps(res, indent=2))
        print(f"S1: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 2

    manifest, n_fail, consecutive_fail = [], 0, 0
    for k, i in enumerate(idx):
        item_id = f"{int(tuples[i]):07d}"
        path = outdir / f"{item_id}.npz"
        if path.exists():
            continue
        gen = generate_one(model, vae, frames[i][:ctxn], actions[i], dev)
        seq = np.stack([frames[i][0], gen])
        checks = validate_item(seq, frames[i][0], expected_count=2)
        digest = hashlib.sha256(gen.tobytes()).hexdigest()
        provenance.save_arrays(path, generated=gen, conditioning=frames[i][:ctxn],
                               action=actions[i], real_target=frames[i][2])
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
