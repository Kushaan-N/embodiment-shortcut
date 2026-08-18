"""Corpus construction, splits, and loading.

The one thing in this file that must not be got wrong is the split (T7).

Rollouts share ``(action, seed)`` across conditions.  If the split unit were
the individual rollout, an INTERACT rollout in train and its DECOY twin in test
would share the exact arm trajectory and the exact action -- near-memorisation
that inflates DECOY test accuracy and biases G **toward the hypothesis**.  The
split unit is therefore the ``(geometry, tuple_index)`` key, and every
condition-variant of a tuple lands in the same split by construction:
``split_for_tuple`` does not take the condition as an argument, so a leak of
this kind is not expressible through this API.

Splits are a deterministic hash, not a shuffle, so the corpus can be extended
with new shards later without re-simulating or re-assigning existing tuples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import config as C
import provenance
import scene

__all__ = [
    "split_for_tuple", "split_counts", "assert_no_tuple_spans_splits",
    "shard_path", "build_shard", "iter_records", "load_split",
    "HORIZONS", "record_from_rollout",
]

HORIZONS = ("s_0", "s_std", "s_del", "s_time")
SPLIT_NAMES = ("train", "val", "test")


# ==========================================================================
# Splits (T7)
# ==========================================================================


def _uniform_hash(*parts) -> float:
    """Deterministic uniform [0, 1) from the given key parts."""
    key = "|".join(str(p) for p in parts).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    (val,) = struct.unpack("<Q", digest)
    return val / 2 ** 64


def split_for_tuple(geometry: str, tuple_index: int, salt: str = C.SPLIT_SALT) -> str:
    """Split assignment for one ``(action, seed)`` tuple.

    Deliberately takes no ``condition`` argument: all condition-variants of a
    tuple must land in the same split (T7), and the cleanest way to guarantee
    that is to make the alternative unrepresentable.
    """
    u = _uniform_hash(salt, geometry, int(tuple_index))
    train_f, val_f, _ = C.SPLIT_FRACTIONS
    if u < train_f:
        return "train"
    if u < train_f + val_f:
        return "val"
    return "test"


def split_counts(geometry: str, n_tuples: int) -> dict:
    c = Counter(split_for_tuple(geometry, i) for i in range(n_tuples))
    return {k: c.get(k, 0) for k in SPLIT_NAMES}


def assert_no_tuple_spans_splits(records) -> dict:
    """ASSERT (T7): no ``(geometry, tuple_index)`` key appears in two splits.

    Checked against the *materialised* records, not just the hash function, so
    that a corpus assembled by any route is still verified.
    """
    seen: dict = defaultdict(set)
    for r in records:
        seen[(r["geometry"], int(r["tuple_index"]))].add(r["split"])
    offenders = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    return {
        "name": "T7_no_tuple_spans_splits",
        "passed": not offenders,
        "value": float(len(offenders)),
        "threshold": 0.0,
        "n_tuples_checked": len(seen),
        "offenders": [list(k) + [v] for k, v in list(offenders.items())[:20]],
        "note": "split unit is (geometry, tuple_index); NEVER the rollout, NEVER the frame",
    }


# ==========================================================================
# Corpus records
# ==========================================================================


def record_from_rollout(r: dict, tuple_index: int, split: str, stride: int | None = None) -> dict:
    """Flatten one rollout into the arrays persisted for it.

    State trajectories are stored at the *video* frame stride rather than the
    500 Hz physics rate: a full-rate trajectory is ~25x larger and no
    downstream experiment consumes it (Experiment A, the only full-rate
    consumer, simulates its own rollouts).  The exact horizon states are stored
    separately at full precision so nothing that matters is interpolated.
    """
    stride = stride or C.FRAME_STRIDE
    hz = r["horizon_idx"]
    sub = slice(0, None, stride)

    out = {
        "action": r["action"].astype(np.float32),
        "seed": np.int64(r["seed"]),
        "tuple_index": np.int64(tuple_index),
        "friction_mult": np.float32(r["friction_mult"]),
        "contact_end": np.int64(r["contact_end"]),
        "settled_at": np.int64(-1 if r["settled_at"] is None else r["settled_at"]),
        "arm_rest_at": np.int64(r["arm_rest_at"]),
        "horizon_indices": np.array([hz[h] for h in HORIZONS], dtype=np.int64),
        "arm_qpos_sub": r["arm_qpos"][sub].astype(np.float32),
        "arm_qpos_horizons": np.stack(
            [r["arm_qpos"][hz[h]] if h != "s_time" else
             (r["arm_qpos_time"][hz[h]] if r["arm_qpos_time"] is not None
              else r["arm_qpos"][hz[h]]) for h in HORIZONS]
        ).astype(np.float32),
        "contact_forces_sub": r["contact_forces"][sub].astype(np.float32),
        "contact_force_peak": np.float32(np.max(r["contact_forces"])
                                         if r["contact_forces"].size else 0.0),
        "frames": np.stack([r["frames"][h] for h in HORIZONS]).astype(np.uint8),
        "seg_masks": np.stack([r["seg_masks"][h] for h in HORIZONS]).astype(np.uint8),
        "validators_passed": np.bool_(r["validators"]["_all_passed"]),
    }
    if r["obj_poses"] is not None:
        out["obj_poses_sub"] = r["obj_poses"][sub].astype(np.float32)
        out["obj_vel_sub"] = r["obj_vel"][sub].astype(np.float32)
        out["obj_poses_horizons"] = np.stack(
            [r["obj_poses"][hz[h]] if h != "s_time" else
             (r["obj_poses_time"][hz[h]] if r["obj_poses_time"] is not None
              else r["obj_poses"][hz[h]]) for h in HORIZONS]
        ).astype(np.float32)
        out["obj_init"] = r["obj_init"].astype(np.float32)
    if r["decoy_xy"] is not None:
        out["decoy_xy"] = r["decoy_xy"].astype(np.float32)
    return out


def shard_path(root: Path, geometry: str, condition: str, shard: int,
               friction_mult: float = 1.0) -> Path:
    """Per-shard filename.

    Modal volume writes race across parallel containers (§5, §14), so every
    container writes its own file and a single-container pass merges them.  No
    two shards ever name the same path.
    """
    fm = "" if abs(friction_mult - 1.0) < 1e-9 else f"_fm{friction_mult:g}"
    return Path(root) / geometry / f"{condition}{fm}_shard{shard:05d}.npz"


# ==========================================================================
# Generation
# ==========================================================================


def build_shard(geometry: str, shard: int, *, conditions=C.CONDITIONS,
                shard_size: int = C.SHARD_SIZE, root: Path = C.CORPUS_ROOT,
                friction_mult: float = 1.0, thresholds=None,
                clip_root: Path | None = None, master_seed: int = C.MASTER_SEED,
                overwrite: bool = False) -> dict:
    """Simulate and persist one shard of ``(action, seed)`` tuples.

    Writes one ``.npz`` per (geometry, condition, shard).  Atomic
    (tmp -> fsync -> rename) so an interrupted run leaves no half-file, and
    skip-if-exists so re-running is idempotent.
    """
    lo = shard * shard_size
    hi = lo + shard_size
    summary = {"geometry": geometry, "shard": shard, "written": {}, "skipped": {},
               "n_tuples": shard_size, "friction_mult": friction_mult}

    thr_g = None
    if thresholds is not None:
        thr_g = thresholds.for_geometry(geometry) if hasattr(thresholds, "for_geometry") \
            else thresholds

    for condition in conditions:
        out_path = shard_path(root, geometry, condition, shard, friction_mult)
        if out_path.exists() and not overwrite:
            summary["skipped"][condition] = str(out_path)
            continue
        per_key: dict = defaultdict(list)
        meta_rows = []
        for ti in range(lo, hi):
            streams = scene.tuple_streams(master_seed, geometry, ti)
            action = scene.sample_action(streams["action"])
            clip_variant = "A-std" if clip_root is not None else None
            r = scene.rollout(action, seed=ti, condition=condition, geometry=geometry,
                              thresholds=thr_g, master_seed=master_seed,
                              friction_mult=friction_mult, clip_variant=clip_variant)
            split = split_for_tuple(geometry, ti)
            rec = record_from_rollout(r, ti, split)
            for k, v in rec.items():
                per_key[k].append(v)
            meta_rows.append({
                "tuple_index": ti, "split": split, "condition": condition,
                "geometry": geometry,
                "validators": {k: v for k, v in r["validators"].items()
                               if isinstance(v, dict)},
            })
            if clip_root is not None and r["clip"] is not None:
                provenance.save_arrays(
                    Path(clip_root) / geometry / condition / f"{ti:07d}.npz",
                    clip=r["clip"].astype(np.uint8),
                    clip_indices=r["clip_indices"].astype(np.int64),
                )

        arrays = {}
        for k, vals in per_key.items():
            arrays[k] = np.stack(vals) if np.ndim(vals[0]) > 0 else np.asarray(vals)
        arrays["split"] = np.asarray([m["split"] for m in meta_rows])
        arrays["condition"] = np.asarray([condition] * len(meta_rows))
        arrays["geometry"] = np.asarray([geometry] * len(meta_rows))

        provenance.save_arrays(out_path, **arrays)
        with open(out_path.with_suffix(".validators.json"), "w") as fh:
            json.dump(provenance._jsonable(meta_rows), fh, indent=1)
        summary["written"][condition] = str(out_path)
    return summary


# ==========================================================================
# Loading
# ==========================================================================


def iter_records(root: Path = C.CORPUS_ROOT, *, geometry=None, condition=None,
                 friction_mult: float = 1.0):
    """Yield (path, npz) for every shard matching the filters."""
    root = Path(root)
    geoms = [geometry] if geometry else list(C.GEOMETRIES)
    conds = [condition] if condition else list(C.CONDITIONS)
    for g in geoms:
        gdir = root / g
        if not gdir.exists():
            continue
        for c in conds:
            fm = "" if abs(friction_mult - 1.0) < 1e-9 else f"_fm{friction_mult:g}"
            for p in sorted(gdir.glob(f"{c}{fm}_shard*.npz")):
                yield p, np.load(p, allow_pickle=False)


def load_split(split: str, *, root: Path = C.CORPUS_ROOT, geometry=None,
               condition=None, friction_mult: float = 1.0,
               keys=None, only_valid: bool = True) -> dict:
    """Concatenate every shard, keeping rows in the requested split.

    ``only_valid`` applies the pre-registered inclusion policy: rollouts whose
    validators failed (INTERACT misses, DECOY grazes, non-settlers,
    out-of-frame) are excluded.  The counts are returned so the exclusion is
    reported rather than silent.
    """
    if split not in SPLIT_NAMES + ("all",):
        raise ValueError(split)
    buckets: dict = defaultdict(list)
    n_seen = n_kept = n_excluded_split = n_excluded_invalid = 0

    for _, z in iter_records(root, geometry=geometry, condition=condition,
                             friction_mult=friction_mult):
        splits = z["split"].astype(str)
        valid = z["validators_passed"].astype(bool)
        mask = np.ones(splits.shape, dtype=bool) if split == "all" else (splits == split)
        n_seen += mask.size
        n_excluded_split += int((~mask).sum())
        if only_valid:
            n_excluded_invalid += int((mask & ~valid).sum())
            mask = mask & valid
        if not mask.any():
            continue
        n_kept += int(mask.sum())
        for k in (keys or z.files):
            if k not in z.files:
                continue
            arr = z[k]
            if arr.shape[:1] != splits.shape:
                continue
            buckets[k].append(arr[mask])

    out = {k: np.concatenate(v, axis=0) for k, v in buckets.items()}
    out["_counts"] = {
        "seen": n_seen, "kept": n_kept,
        "excluded_other_split": n_excluded_split,
        "excluded_failed_validators": n_excluded_invalid,
    }
    return out


# ==========================================================================
# CLI
# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate corpus shards.")
    ap.add_argument("--geometry", choices=C.GEOMETRIES, required=True)
    ap.add_argument("--shards", type=int, nargs="+", required=True,
                    help="shard indices to build (per-shard filenames; safe to parallelise)")
    ap.add_argument("--shard-size", type=int, default=C.SHARD_SIZE)
    ap.add_argument("--friction-mult", type=float, default=1.0)
    ap.add_argument("--root", type=Path, default=C.CORPUS_ROOT)
    ap.add_argument("--clips", action="store_true", help="also render and store A-std clips")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    thresholds = C.load_thresholds()  # raises if Experiment A has not run
    clip_root = C.CLIP_ROOT if args.clips else None
    for s in args.shards:
        info = build_shard(args.geometry, s, shard_size=args.shard_size,
                           root=args.root, friction_mult=args.friction_mult,
                           thresholds=thresholds, clip_root=clip_root,
                           overwrite=args.overwrite)
        print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
