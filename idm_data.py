"""Torch data plumbing for the IDM variants (§8.1).

Three input regimes, pinned per metric so T8 cannot bite:

  pair  -- ``(s_0, s_h)`` only.  A-del, A-time, and every Architecture B probe.
           Excluding intermediate frames is *part of the definition* of OG-AF.
  clip  -- the frames of ``clip_indices(variant)``.  A-std (the faithful
           reproduction, whose IDM is bidirectional per §B.2) and A-clip-del
           (the ablation that shows clip access re-opens the shortcut).
  embed -- pre-computed frozen-encoder features, for Architecture B.

Frames are materialised once into uint8 memmaps: the corpus is ~11 GB of
frames, which does not fit in RAM but memmaps fine, and re-decompressing shards
inside a shuffled DataLoader would dominate training time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config as C
import datasets as ds
import idm

__all__ = [
    "materialise_frames", "FrameStore", "PairDataset", "ClipDataset",
    "EmbeddingDataset", "make_loader", "MASK_MODES", "masked_frames",
]

MASK_MODES = ("full", "arm_masked", "object_masked")
_MASK_KEEP = {"arm_masked": C.SEG_OBJECT, "object_masked": C.SEG_ARM}


# ==========================================================================
# Materialised frame store
# ==========================================================================


def _store_paths(root: Path, geometry: str, condition: str, friction_mult: float):
    fm = "" if abs(friction_mult - 1.0) < 1e-9 else f"_fm{friction_mult:g}"
    base = Path(root) / f"{geometry}_{condition}{fm}"
    return base.with_suffix(".frames.npy"), base.with_suffix(".masks.npy"), \
        base.with_suffix(".index.json")


def materialise_frames(geometry: str, condition: str, *, corpus_root: Path = C.CORPUS_ROOT,
                       store_root: Path | None = None, friction_mult: float = 1.0,
                       overwrite: bool = False) -> dict:
    """Flatten shards into one uint8 memmap of horizon frames + masks."""
    store_root = Path(store_root or (C.DATA_ROOT / "frame_store"))
    store_root.mkdir(parents=True, exist_ok=True)
    fpath, mpath, ipath = _store_paths(store_root, geometry, condition, friction_mult)
    if ipath.exists() and not overwrite:
        return json.loads(ipath.read_text())

    rows = []
    frames_list, masks_list = [], []
    for path, z in ds.iter_records(corpus_root, geometry=geometry, condition=condition,
                                   friction_mult=friction_mult):
        n = len(z["tuple_index"])
        frames_list.append(z["frames"])
        masks_list.append(z["seg_masks"])
        for i in range(n):
            rows.append({
                "tuple_index": int(z["tuple_index"][i]),
                "split": str(z["split"][i]),
                "action": z["action"][i].tolist(),
                "valid": bool(z["validators_passed"][i]),
                "shard": path.name,
            })
    if not rows:
        raise FileNotFoundError(
            f"no corpus shards for {geometry}/{condition} under {corpus_root}"
        )

    frames = np.concatenate(frames_list)     # (N, 4, H, W, 3)
    masks = np.concatenate(masks_list)       # (N, 4, H, W)
    np.save(fpath, frames)
    np.save(mpath, masks)
    index = {"geometry": geometry, "condition": condition, "friction_mult": friction_mult,
             "n": len(rows), "horizons": list(ds.HORIZONS), "rows": rows,
             "frames_path": str(fpath), "masks_path": str(mpath)}
    ipath.write_text(json.dumps(index))
    return index


class FrameStore:
    """Read-only view over a materialised (frames, masks, index) triple."""

    def __init__(self, geometry: str, condition: str, *, store_root: Path | None = None,
                 friction_mult: float = 1.0):
        store_root = Path(store_root or (C.DATA_ROOT / "frame_store"))
        fpath, mpath, ipath = _store_paths(store_root, geometry, condition, friction_mult)
        if not ipath.exists():
            raise FileNotFoundError(f"{ipath} missing; run materialise_frames first")
        self.index = json.loads(ipath.read_text())
        self.frames = np.load(fpath, mmap_mode="r")
        self.masks = np.load(mpath, mmap_mode="r")
        self.geometry, self.condition = geometry, condition
        self.horizon_pos = {h: i for i, h in enumerate(self.index["horizons"])}

    def rows_for_split(self, split: str, only_valid: bool = True):
        out = []
        for i, r in enumerate(self.index["rows"]):
            if split != "all" and r["split"] != split:
                continue
            if only_valid and not r["valid"]:
                continue
            out.append(i)
        return out


def masked_frames(frames: np.ndarray, masks: np.ndarray, mode: str,
                  background: np.ndarray) -> np.ndarray:
    """Apply the §8.4 masking, filling with the background render."""
    if mode == "full":
        return frames
    if mode not in _MASK_KEEP:
        raise ValueError(f"unknown mask mode {mode!r}")
    return idm.composite_mask(frames, masks, background, _MASK_KEEP[mode])


# ==========================================================================
# Datasets
# ==========================================================================


class _Base(Dataset):
    def __init__(self, stores, split, *, mask_mode="full", background=None,
                 only_valid=True):
        self.items = []       # (store_idx, row_idx)
        self.stores = list(stores)
        self.mask_mode = mask_mode
        self.background = background
        if mask_mode != "full" and background is None:
            raise ValueError("masking needs the background render (§8.4)")
        for si, st in enumerate(self.stores):
            for ri in st.rows_for_split(split, only_valid=only_valid):
                self.items.append((si, ri))

    def __len__(self):
        return len(self.items)

    def meta(self) -> dict:
        """Per-sample identity, for the paired analysis of §7.4."""
        tup, cond, geom, act = [], [], [], []
        for si, ri in self.items:
            st = self.stores[si]
            row = st.index["rows"][ri]
            tup.append(row["tuple_index"])
            cond.append(st.condition)
            geom.append(st.geometry)
            act.append(row["action"])
        return {"tuple_index": np.asarray(tup), "condition": np.asarray(cond),
                "geometry": np.asarray(geom), "action": np.asarray(act, dtype=np.float32)}

    def _target(self, si, ri):
        row = self.stores[si].index["rows"][ri]
        return torch.tensor(idm.normalise_action(row["action"]), dtype=torch.float32)

    def _frames(self, si, ri, positions):
        st = self.stores[si]
        f = np.asarray(st.frames[ri][positions])          # (T, H, W, 3) uint8
        if self.mask_mode != "full":
            m = np.asarray(st.masks[ri][positions])
            f = masked_frames(f, m, self.mask_mode, self.background)
        x = torch.from_numpy(np.ascontiguousarray(f)).permute(0, 3, 1, 2)
        return x.contiguous().float() / 255.0


class PairDataset(_Base):
    """``(s_0, s_h)`` frame pairs -- the OG-AF input regime (§8.1, T8)."""

    def __init__(self, stores, split, horizon: str, **kw):
        super().__init__(stores, split, **kw)
        self.horizon = horizon
        self.positions = [self.stores[0].horizon_pos["s_0"],
                          self.stores[0].horizon_pos[horizon]]

    def __getitem__(self, i):
        si, ri = self.items[i]
        return self._frames(si, ri, self.positions), self._target(si, ri)


class ClipDataset(_Base):
    """Clip input for A-std / A-clip-del, read lazily from per-rollout files."""

    def __init__(self, stores, split, variant: str, clip_root: Path = C.CLIP_ROOT, **kw):
        super().__init__(stores, split, **kw)
        if kw.get("mask_mode", "full") != "full":
            raise ValueError("masking decomposition is Architecture B / pair-input only (§8.4)")
        self.variant = variant
        self.clip_root = Path(clip_root)

    def __getitem__(self, i):
        si, ri = self.items[i]
        st = self.stores[si]
        row = st.index["rows"][ri]
        path = self.clip_root / st.geometry / st.condition / f"{row['tuple_index']:07d}.npz"
        with np.load(path) as z:
            clip = z["clip"]
        x = torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2)
        return x.contiguous().float() / 255.0, self._target(si, ri)


class EmbeddingDataset(Dataset):
    """Pre-computed ``[embed(s_0), embed(s_h)]`` pairs for Architecture B."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, meta: dict):
        self.x = torch.from_numpy(np.ascontiguousarray(features)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(targets)).float()
        self._meta = meta

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def meta(self):
        return self._meta


def make_loader(dataset, batch_size: int, *, shuffle: bool, seed: int = 0,
                num_workers: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=False, persistent_workers=bool(num_workers),
    )
