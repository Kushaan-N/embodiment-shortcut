"""Train one IDM: `(variant, seed)` -> checkpoint + held-out per-sample errors.

Substrate-agnostic: pure PyTorch, paths from environment variables, no Modal
imports.  The identical file runs locally, on Modal, and under Unity sbatch.

**One IDM per variant, never per condition** (§8.2).  The whole point is a
single model evaluated across conditions; a per-condition model could not
exhibit the confound at all.

Held-out evaluation defaults to **INTERACT only**.  Experiment D's floors are
INTERACT-only by design (§9-D), and per-condition contrasts -- anything that
reveals G -- stay sealed until `prereg.md` is committed and pushed.  Producing
them requires `--eval-conditions all`, which `exp_e_confound.py` invokes after
the §0.5 lock has been checked.

    python train_idm.py --variant A-del --seed 0
    python train_idm.py --variant B-std --seed 3 --encoder facebook/dinov2-large
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import config as C
import datasets as ds
import idm
import idm_data as idd
import protocol as proto_mod
import provenance
import scene

#: Gate B selected this; see config.GATE_B_EVIDENCE for the measurement.
DEFAULT_ENCODER = C.GATE_B_SELECTED_ENCODER


# ==========================================================================


def build_stores(geometries, conditions, friction_mult=1.0):
    stores = []
    for g in geometries:
        for c in conditions:
            stores.append(idd.FrameStore(g, c, friction_mult=friction_mult))
    return stores


def embed_stores(stores, split, horizon, encoder_name, *, mask_mode="full",
                 background=None, cache_root: Path | None = None, device=None):
    """Frozen-encoder features for Architecture B / the masking decomposition.

    Cached per (encoder, geometry, condition, horizon, mask_mode) so that the
    10 Architecture B seeds and the 5 masking seeds share one encoder pass.
    """
    import exp_b_resolution as expb

    cache_root = Path(cache_root or (C.DATA_ROOT / "embeddings"))
    cache_root.mkdir(parents=True, exist_ok=True)
    device = device or idm.device()
    enc = None
    feats, targets = [], []
    meta = {"tuple_index": [], "condition": [], "geometry": [], "action": []}

    for st in stores:
        fm = getattr(st, "friction_mult", 1.0)
        fm_tag = "" if abs(fm - 1.0) < 1e-9 else f"__fm{fm:g}"
        # The friction multiplier MUST be in the key: Experiment F evaluates
        # the same (geometry, condition, horizon, split) at several
        # multipliers, and a shared cache would score the x1.0 embeddings at
        # every multiplier -- a flat dose-response by construction.
        tag = (f"{encoder_name.replace('/', '_')}__{st.geometry}__{st.condition}"
               f"{fm_tag}__{horizon}__{mask_mode}__{split}")
        cache = cache_root / f"{tag}.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            feats.append(z["features"])
            targets.append(z["targets"])
            for k in meta:
                meta[k].append(z[k])
            continue

        rows = st.rows_for_split(split)
        if not rows:
            continue
        if enc is None:
            enc = expb.FrozenEncoder(encoder_name, device)
        pos = [st.horizon_pos["s_0"], st.horizon_pos[horizon]]
        chunk_f, chunk_t = [], []
        m_local = {k: [] for k in meta}
        for start in range(0, len(rows), 64):
            batch_rows = rows[start : start + 64]
            # np.ix_ reads ONLY the two horizons wanted.  `frames[rows][:, pos]`
            # materialises all four first and throws half away -- measured 2.7x
            # slower on a 1.1 GB store, and the corpus store is ~11 GB.  The
            # selected pixels are identical, so this is a pure I/O change.
            sel = np.ix_(batch_rows, pos)
            imgs = np.asarray(st.frames[sel])                         # (B, 2, H, W, 3)
            if mask_mode != "full":
                msk = np.asarray(st.masks[sel])
                imgs = idd.masked_frames(imgs, msk, mask_mode, background)
            B = imgs.shape[0]
            flat = imgs.reshape(B * 2, *imgs.shape[2:])
            e = enc(flat)
            pooled = np.concatenate([e["cls"], e["mean_patch"]], axis=-1)
            chunk_f.append(pooled.reshape(B, -1))
            for ri in batch_rows:
                r = st.index["rows"][ri]
                chunk_t.append(idm.normalise_action(r["action"]))
                m_local["tuple_index"].append(r["tuple_index"])
                m_local["condition"].append(st.condition)
                m_local["geometry"].append(st.geometry)
                m_local["action"].append(r["action"])
        F = np.concatenate(chunk_f).astype(np.float32)
        T = np.asarray(chunk_t, dtype=np.float32)
        payload = {"features": F, "targets": T,
                   **{k: np.asarray(v) for k, v in m_local.items()}}
        provenance.save_arrays(cache, **payload)
        feats.append(F)
        targets.append(T)
        for k in meta:
            meta[k].append(payload[k])

    if not feats:
        raise RuntimeError(f"no data for split={split}")
    return (np.concatenate(feats), np.concatenate(targets),
            {k: np.concatenate(v) for k, v in meta.items()})


# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=sorted(C.IDM_VARIANTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--geometries", nargs="+", default=list(C.GEOMETRIES))
    ap.add_argument("--train-conditions", nargs="+", default=list(C.CONDITIONS),
                    help="mixed-condition training so no condition is OOD at test (T2)")
    ap.add_argument("--eval-conditions", nargs="+", default=["INTERACT"],
                    help="INTERACT only until prereg.md is committed (§9-D)")
    ap.add_argument("--mask-mode", default="full", choices=idd.MASK_MODES)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--augment", action="store_true", help="§8.3 appearance augmentation")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=C.CHECKPOINT_ROOT)
    ap.add_argument("--friction-mult", type=float, default=1.0)
    args = ap.parse_args()

    t0 = time.time()
    proto = proto_mod.load()
    print(proto.summary_table(), flush=True)

    horizon = C.IDM_VARIANTS[args.variant]
    is_clip = args.variant in C.CLIP_VARIANTS
    is_arch_a = args.variant.startswith("A-")
    dev = idm.device()
    tag = (f"{args.variant}__seed{args.seed}__{args.mask_mode}"
           f"{'__aug' if args.augment else ''}")
    outdir = Path(args.out) / tag
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {tag}  horizon={horizon}  clip={is_clip}  device={dev} ===", flush=True)

    if args.mask_mode != "full" and is_arch_a:
        raise SystemExit("§8.4: the masking decomposition is Architecture B only")

    background = scene.render_empty_scene(args.geometries[0]) \
        if args.mask_mode != "full" else None

    train_stores = build_stores(args.geometries, args.train_conditions, args.friction_mult)
    mixture = {f"{s.geometry}/{s.condition}": len(s.rows_for_split("train"))
               for s in train_stores}
    print(f"training mixture (§8.2): {json.dumps(mixture)}", flush=True)

    # Seed BEFORE the model is built.  idm.train_idm seeds again before the
    # loop, but by then build_model has already drawn the initial weights from
    # the process-default generator, which is the SAME for every seed -- the
    # 5/10 seeds would differ only in shuffle order, not initialisation, and
    # would not be the independent replicates the seed-level summary assumes.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------------- build model + loaders --------------------------------
    if is_arch_a:
        cfg = idm.train_config_from_protocol(proto, args.variant, args.seed,
                                             augment=args.augment, epochs=args.epochs)
        if args.batch_size:
            cfg.batch_size = args.batch_size
        cfg.num_workers = args.num_workers
        model = idm.build_model(args.variant, proto)
        if is_clip:
            mk = lambda sp: idd.ClipDataset(train_stores, sp, args.variant)  # noqa: E731
        else:
            mk = lambda sp: idd.PairDataset(train_stores, sp, horizon)       # noqa: E731
        tr_ds, va_ds = mk("train"), mk("val")
        tr = idd.make_loader(tr_ds, cfg.batch_size, shuffle=True, seed=args.seed,
                             num_workers=cfg.num_workers)
        # Validation is a small pass; 8 persistent workers on top of the 8
        # training workers exceeds the 8-CPU allocation for nothing.
        va = idd.make_loader(va_ds, cfg.batch_size, shuffle=False,
                             num_workers=min(2, cfg.num_workers))
    else:
        Xtr, Ytr, _ = embed_stores(train_stores, "train", horizon, args.encoder,
                                   mask_mode=args.mask_mode, background=background)
        Xva, Yva, _ = embed_stores(train_stores, "val", horizon, args.encoder,
                                   mask_mode=args.mask_mode, background=background)
        mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
        Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
        cfg = idm.TrainConfig(variant=args.variant, seed=args.seed,
                              epochs=args.epochs or 150, lr=1e-3,
                              batch_size=args.batch_size or 256, weight_decay=1e-2,
                              optimizer="adamw", lr_schedule="cosine_with_warmup",
                              augment=False, num_workers=0, amp=False)
        model = idm.build_model(args.variant, d_in=Xtr.shape[1])
        tr = idd.make_loader(idd.EmbeddingDataset(Xtr, Ytr, {}), cfg.batch_size,
                             shuffle=True, seed=args.seed)
        va = idd.make_loader(idd.EmbeddingDataset(Xva, Yva, {}), cfg.batch_size,
                             shuffle=False)

    aug = idm.AppearanceAugment(enabled=args.augment) if (is_arch_a and args.augment) else None
    history = idm.train_idm(model, tr, va, cfg, dev=dev, augment=aug)

    ck_path = outdir / "checkpoint.pt"
    ck_tmp = ck_path.with_suffix(".pt.tmp")
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.to_dict(),
                "variant": args.variant, "seed": args.seed,
                "mask_mode": args.mask_mode, "encoder": args.encoder,
                "protocol_sha256": proto.sha256,
                "embed_norm": None if is_arch_a else {"mu": mu, "sd": sd}},
               ck_tmp)
    os.replace(ck_tmp, ck_path)   # atomic: never a half-written checkpoint

    # ---------------- held-out evaluation ---------------------------------
    evals = {}
    for cond in args.eval_conditions:
        stores = build_stores(args.geometries, [cond], args.friction_mult)
        if is_arch_a:
            d = (idd.ClipDataset(stores, "test", args.variant) if is_clip
                 else idd.PairDataset(stores, "test", horizon))
            loader = idd.make_loader(d, cfg.batch_size, shuffle=False,
                                     num_workers=min(2, cfg.num_workers))
            meta = d.meta()
        else:
            Xte, Yte, meta = embed_stores(stores, "test", horizon, args.encoder,
                                          mask_mode=args.mask_mode, background=background)
            Xte = (Xte - mu) / sd
            loader = idd.make_loader(idd.EmbeddingDataset(Xte, Yte, meta),
                                     cfg.batch_size, shuffle=False)
        res = idm.evaluate_idm(model, loader, dev=dev, return_per_sample=True)
        evals[cond] = {"res": res, "meta": meta}
        print(f"  held-out {cond:9s} n={res['n']:5d}  MAE={res['mae']:.5f}  "
              f"MSE={res['mse']:.5f}  (prior MAE = 0.25)", flush=True)

    arrays = {}
    for cond, e in evals.items():
        arrays[f"{cond}_abs_err"] = e["res"]["abs_err"]
        arrays[f"{cond}_sq_err"] = e["res"]["sq_err"]
        arrays[f"{cond}_pred"] = e["res"]["pred"]
        arrays[f"{cond}_target"] = e["res"]["target"]
        arrays[f"{cond}_tuple_index"] = e["meta"]["tuple_index"]
        arrays[f"{cond}_geometry"] = e["meta"]["geometry"]
    for k, v in history.items():
        arrays[f"history_{k}"] = np.asarray(v)

    provenance.save_run(
        outdir, f"train_idm::{tag}", arrays,
        seeds={"train_seed": args.seed, "master": C.MASTER_SEED},
        extra={"variant": args.variant, "horizon": horizon, "is_clip": is_clip,
               "mask_mode": args.mask_mode, "encoder": args.encoder,
               "augment": args.augment, "train_config": cfg.to_dict(),
               "training_mixture": mixture, "eval_conditions": args.eval_conditions,
               "protocol": proto.to_dict(), "wall_clock_s": time.time() - t0,
               "summary": {c: {k: v for k, v in e["res"].items()
                               if not isinstance(v, np.ndarray)}
                           for c, e in evals.items()}},
        arrays_name="eval.npz",
    )
    print(f"wrote {outdir}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
