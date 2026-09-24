"""Render video-rate clips for the Experiment H world model (§9-H).

The world model under test is a **video** predictor, so it needs video, not the
four horizon frames the IDM corpus stores.

Crucially this does **not** re-simulate.  ``datasets.record_from_rollout``
already persists ``arm_qpos_sub`` and ``obj_poses_sub`` -- the complete scene
state at every video frame -- so the clip can be reconstructed by posing the
model at each stored state and rendering.  Re-simulating would risk producing
video that does not correspond to the corpus it is paired with (a different
MuJoCo build, a changed constant), and no validator would catch it.

Clips span ``[0, s_del]``: the world model is asked to generate the interval
that ends in the settled contact outcome, which is exactly what OG-AF scores.

    python wm/render_clips.py --geometry box --shards 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mj_env  # noqa: F401,E402  -- must precede `import mujoco` (§14)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import config as C  # noqa: E402
import datasets as ds  # noqa: E402
import provenance  # noqa: E402
import scene  # noqa: E402

#: Frames per world-model clip.  Spans [0, s_del] uniformly.
WM_CLIP_FRAMES = 16


def wm_clip_frame_indices(s_del_step: int, n_frames: int = WM_CLIP_FRAMES,
                          stride: int = None) -> np.ndarray:
    """Sub-sampled *video frame* indices spanning [0, s_del].

    Returned in units of the stored trajectory rows (video frames), not
    physics steps.
    """
    stride = stride or C.FRAME_STRIDE
    last_frame = int(s_del_step) // stride
    if last_frame < 1:
        raise ValueError(f"s_del at step {s_del_step} is before the first video frame")
    return np.unique(np.linspace(0, last_frame, n_frames).round().astype(np.int64))


def render_clip(sm, arm_traj: np.ndarray, obj_traj: np.ndarray | None,
                frame_idx: np.ndarray, renderer) -> np.ndarray:
    """Pose the model at each stored state and render.  (T, H, W, 3) uint8."""
    data = mujoco.MjData(sm.model)
    out = []
    for f in frame_idx:
        mujoco.mj_resetData(sm.model, data)
        data.qpos[sm.arm_qadr] = arm_traj[int(f)]
        if sm.obj_qadr is not None:
            if obj_traj is None:
                raise ValueError("condition has an object but no trajectory was given")
            data.qpos[sm.obj_qadr : sm.obj_qadr + 7] = obj_traj[int(f)]
        mujoco.mj_forward(sm.model, data)
        renderer.update_scene(data, camera=C.CAMERA_NAME)
        out.append(renderer.render().copy())
    return np.stack(out)


def process_shard(geometry: str, shard_path: Path, out_root: Path,
                  n_frames: int = WM_CLIP_FRAMES, condition: str = "INTERACT",
                  overwrite: bool = False) -> dict:
    sm = scene.get_scene(geometry, condition, 1.0)
    z = np.load(shard_path, allow_pickle=False)
    tuples = z["tuple_index"]
    arm = z["arm_qpos_sub"]
    obj = z["obj_poses_sub"] if "obj_poses_sub" in z.files else None
    hz = z["horizon_indices"]          # (N, 4) -> s_0, s_std, s_del, s_time
    actions = z["action"]
    splits = z["split"].astype(str)
    valid = z["validators_passed"].astype(bool)

    written = skipped = failed = 0
    with mujoco.Renderer(sm.model, height=C.RENDER_SIZE, width=C.RENDER_SIZE) as r:
        for i in range(len(tuples)):
            if not valid[i]:
                failed += 1
                continue
            path = Path(out_root) / geometry / condition / f"{int(tuples[i]):07d}.npz"
            if path.exists() and not overwrite:
                skipped += 1
                continue
            try:
                idx = wm_clip_frame_indices(int(hz[i, 2]), n_frames)
            except ValueError:
                failed += 1
                continue
            clip = render_clip(sm, arm[i], None if obj is None else obj[i], idx, r)
            provenance.save_arrays(
                path, clip=clip.astype(np.uint8), frame_indices=idx.astype(np.int64),
                action=actions[i].astype(np.float32),
                tuple_index=np.int64(tuples[i]), split=np.str_(splits[i]),
                s_del_step=np.int64(hz[i, 2]),
            )
            written += 1
    return {"shard": shard_path.name, "written": written, "skipped": skipped,
            "excluded": failed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", choices=C.GEOMETRIES, required=True)
    ap.add_argument("--shards", type=int, nargs="+", required=True)
    ap.add_argument("--condition", default="INTERACT", choices=C.CONDITIONS)
    ap.add_argument("--frames", type=int, default=WM_CLIP_FRAMES)
    ap.add_argument("--corpus", type=Path, default=C.CORPUS_ROOT)
    ap.add_argument("--out", type=Path, default=C.DATA_ROOT / "wm_clips")
    ap.add_argument("--friction-mult", type=float, default=1.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    fm = "" if abs(args.friction_mult - 1.0) < 1e-9 else f"_fm{args.friction_mult:g}"
    # A friction-swept corpus gets its OWN clip root (the sibling wm_train.py
    # reads: wm_clips_fm3/...).  Writing swept clips into the baseline root
    # would either be skipped as "exists" or overwrite what WM-base trains on.
    out_root = Path(args.out)
    if fm:
        out_root = out_root.parent / f"{out_root.name}{fm}"
    missing, written, skipped = [], 0, 0
    for s in args.shards:
        p = Path(args.corpus) / args.geometry / f"{args.condition}{fm}_shard{s:05d}.npz"
        if not p.exists():
            print(f"  MISSING shard {p}", flush=True)
            missing.append(str(p))
            continue
        info = process_shard(args.geometry, p, out_root, args.frames,
                             args.condition, args.overwrite)
        written += info["written"]
        skipped += info["skipped"]
        print(f"  {args.geometry} shard {s}: {info}", flush=True)
    # Content, not exit code (§13.5): a run that found nothing to render must
    # not exit 0 and leave wm_train.py to starve after a GPU is allocated.
    if missing:
        print(f"{len(missing)} requested shard(s) missing under {args.corpus}; "
              f"build them first (datasets.py --friction-mult ...)", flush=True)
        return 2
    if written + skipped == 0:
        print("nothing rendered and nothing already present -- corpus empty?", flush=True)
        return 2
    print(f"clips under {out_root}: {written} written, {skipped} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
