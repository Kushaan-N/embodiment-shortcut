"""Action-conditioned **video** world model for Experiment H (§9-H).

WM-base: a frozen compact VAE plus a factorised spatiotemporal transformer that
predicts latent frames **autoregressively**, conditioned on ``a in R^3`` via
AdaLN.

Why factorised attention.  Joint attention over all 16x196 = 3136 latent tokens
is quadratic and would dominate the FLOP budget for no modelling benefit at this
scale.  Each block instead does spatial attention *within* a frame (196 tokens,
full) then temporal attention *across* frames (16 tokens, **causally masked**).
Causal masking in time is what makes the model a generator rather than an
interpolator -- without it, "predicting" frame t could just read frame t+1.

Checkpoint / resume.  Training saves optimizer, scheduler, step and RNG state,
resumes automatically from the newest checkpoint, exits cleanly at a wall-clock
budget, and saves immediately on SIGTERM.  Unity's `gpu-preempt` can kill a job
after 2 hours and `--requeue` then restarts it; without resume that is an
infinite loop that never finishes a run.

    python wm/wm_train.py --model WM-base --max-hours 1.8
    python wm/wm_train.py --model WM-base --budget-only    # FLOPs, no training
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C  # noqa: E402
import provenance  # noqa: E402
from wm.vae import CompactVAE  # noqa: E402

LADDER_PATH = Path(__file__).resolve().parent / "ladder.yaml"


# ==========================================================================
# Model
# ==========================================================================


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SpatioTemporalBlock(nn.Module):
    """Spatial (full) then temporal (causal) attention, both AdaLN-conditioned."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        nrm = lambda: nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)  # noqa: E731
        self.n_s, self.n_t, self.n_m = nrm(), nrm(), nrm()
        self.attn_s = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attn_t = nn.MultiheadAttention(dim, heads, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond, causal_mask):
        # x: (B, T, P, D)
        B, T, P, D = x.shape
        (ss, sc, sg, ts, tc, tg, ms, mc, mg) = self.ada(cond).chunk(9, dim=-1)

        # ---- spatial: attend within each frame -------------------------
        h = x.reshape(B * T, P, D)
        hs = modulate(self.n_s(h), ss.repeat_interleave(T, 0), sc.repeat_interleave(T, 0))
        hs, _ = self.attn_s(hs, hs, hs, need_weights=False)
        h = h + sg.repeat_interleave(T, 0).unsqueeze(1) * hs
        x = h.reshape(B, T, P, D)

        # ---- temporal: attend across frames, CAUSALLY ------------------
        h = x.permute(0, 2, 1, 3).reshape(B * P, T, D)
        ht = modulate(self.n_t(h), ts.repeat_interleave(P, 0), tc.repeat_interleave(P, 0))
        ht, _ = self.attn_t(ht, ht, ht, attn_mask=causal_mask, need_weights=False)
        h = h + tg.repeat_interleave(P, 0).unsqueeze(1) * ht
        x = h.reshape(B, P, T, D).permute(0, 2, 1, 3)

        # ---- mlp -------------------------------------------------------
        h = x.reshape(B * T, P, D)
        hm = modulate(self.n_m(h), ms.repeat_interleave(T, 0), mc.repeat_interleave(T, 0))
        h = h + mg.repeat_interleave(T, 0).unsqueeze(1) * self.mlp(hm)
        return h.reshape(B, T, P, D)


class VideoWorldModel(nn.Module):
    """Given latent frames 0..T-1 and an action, predict frames 1..T."""

    def __init__(self, latent_channels=4, grid=28, patch=2, dim=768, depth=6,
                 heads=12, max_frames=16, n_actions=3):
        super().__init__()
        self.grid, self.patch = grid, patch
        self.latent_channels = latent_channels
        self.n_patch = (grid // patch) ** 2
        in_dim = latent_channels * patch * patch
        self.embed = nn.Linear(in_dim, dim)
        self.pos_s = nn.Parameter(torch.zeros(1, 1, self.n_patch, dim))
        self.pos_t = nn.Parameter(torch.zeros(1, max_frames, 1, dim))
        nn.init.trunc_normal_(self.pos_s, std=0.02)
        nn.init.trunc_normal_(self.pos_t, std=0.02)
        self.action_mlp = nn.Sequential(nn.Linear(n_actions, dim), nn.SiLU(),
                                        nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(
            [SpatioTemporalBlock(dim, heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head = nn.Linear(dim, in_dim)
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _patchify(self, z):
        B, T, Cl, G, _ = z.shape
        p = self.patch
        z = z.reshape(B * T, Cl, G, G)
        z = z.unfold(2, p, p).unfold(3, p, p)          # (BT, Cl, G/p, G/p, p, p)
        z = z.permute(0, 2, 3, 1, 4, 5).reshape(B, T, self.n_patch, -1)
        return z

    def _unpatchify(self, x):
        B, T = x.shape[:2]
        p, g = self.patch, self.grid // self.patch
        x = x.reshape(B * T, g, g, self.latent_channels, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, T, self.latent_channels,
                                                self.grid, self.grid)
        return x

    def forward(self, latents, action):
        """latents: (B, T, Cl, G, G) -> predicted NEXT latent for each t."""
        B, T = latents.shape[:2]
        h = self.embed(self._patchify(latents))
        h = h + self.pos_s + self.pos_t[:, :T]
        cond = self.action_mlp(action)
        mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), 1)
        for blk in self.blocks:
            h = blk(h, cond, mask)
        shift, scale = self.ada_out(cond).chunk(2, dim=-1)
        h = h.reshape(B * T, self.n_patch, -1)
        h = modulate(self.norm(h), shift.repeat_interleave(T, 0),
                     scale.repeat_interleave(T, 0))
        return self._unpatchify(self.head(h).reshape(B, T, self.n_patch, -1))


# ==========================================================================
# Config
# ==========================================================================


def resolve_model(name: str) -> dict:
    doc = yaml.safe_load(LADDER_PATH.read_text())
    base = doc["base"]
    for m in doc["models"]:
        if m["name"] == name or m["name"].startswith(name):
            cfg = json.loads(json.dumps(base))
            for k, v in (m.get("overrides") or {}).items():
                cfg[k].update(v)
            cfg["_model"] = m
            cfg["_evaluation"] = doc["evaluation"]
            return cfg
    raise SystemExit(f"unknown model {name!r}; options: "
                     f"{[m['name'] for m in doc['models']]}")


def flop_budget(model: nn.Module, cfg: dict, n_frames: int, n_patch: int) -> dict:
    """Forward+backward FLOPs per step and the resulting GPU-hour estimate.

    Reported, not assumed: §13.3's S2 stage measures real wall-clock and sets
    S4's --time from it.  This exists so the plan is checkable before spending.
    """
    n = sum(p.numel() for p in model.parameters())
    bs = cfg["train"]["batch_size"]
    steps = cfg["train"]["steps"]
    tokens = n_frames * n_patch
    dim = cfg["transformer"]["dim"]
    depth = cfg["transformer"]["depth"]

    linear = 6 * n * bs * tokens
    attn_s = 12 * depth * dim * (n_patch ** 2) * n_frames * bs
    attn_t = 12 * depth * dim * (n_frames ** 2) * n_patch * bs
    total = linear + attn_s + attn_t
    out = {"params_M": n / 1e6, "tokens_per_sample": tokens,
           "tflop_per_step": total / 1e12,
           "attention_share": (attn_s + attn_t) / total,
           "steps": steps, "batch_size": bs}
    for name, tflops in (("a100", 125.0), ("l40s", 65.0)):
        out[f"gpu_hours_{name}"] = steps * (total / (tflops * 1e12)) / 3600
    return out


# ==========================================================================
# Latent cache
# ==========================================================================


def build_latent_cache(vae: CompactVAE, clip_root: Path, geometries, cache_path: Path,
                       device, fraction: float, seed: int, split: str = "train",
                       overwrite: bool = False) -> tuple:
    """Encode every WM clip once and memmap the result.

    Cached deliberately: a preempted job that had to re-encode the corpus on
    every restart would spend most of its wall-clock budget encoding rather
    than training, and under a 2-hour preemption window it might never make
    progress at all.
    """
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".meta.json")
    if meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text())
        want = {"geometries": list(geometries), "fraction": float(fraction),
                "split": split, "seed": int(seed)}
        have = {k: meta.get(k) for k in want}
        if have == want:
            lat = np.load(cache_path, mmap_mode="r")
            act = np.load(cache_path.with_suffix(".actions.npy"))
            print(f"  latent cache hit: {lat.shape} from {cache_path}")
            return lat, act, meta
        # A cache from an earlier invocation (box-only, other fraction/seed)
        # must never be silently reused: it would train the wrong model.
        print(f"  latent cache at {cache_path} was built for {have}, need {want}; "
              f"rebuilding", flush=True)

    files = []
    for g in geometries:
        files.extend(sorted((Path(clip_root) / g / "INTERACT").glob("*.npz")))
    if not files:
        raise FileNotFoundError(f"no WM clips under {clip_root}; run wm/render_clips.py")

    keep = []
    for f in files:
        with np.load(f) as z:
            if str(z["split"]) == split:
                keep.append(f)
    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(keep), size=max(1, int(len(keep) * fraction)), replace=False)
        keep = [keep[i] for i in sorted(idx)]
    print(f"  encoding {len(keep)} clips (split={split}, fraction={fraction})", flush=True)

    lat_list, act_list = [], []
    with torch.no_grad():
        for i in range(0, len(keep), 8):
            batch, acts = [], []
            for f in keep[i : i + 8]:
                with np.load(f) as z:
                    batch.append(z["clip"])
                    acts.append(z["action"])
            x = torch.from_numpy(np.stack(batch))            # (b, T, H, W, 3)
            b, T = x.shape[:2]
            x = x.reshape(b * T, *x.shape[2:]).permute(0, 3, 1, 2)
            x = x.contiguous().float().to(device) / 255.0
            mu, _ = vae.encode(x)
            # fp32: verify_fidelity certifies the delta_pos_min latent shift in
            # fp32, and fp16 resolution (~1e-3 at |mu|~2) can eat it.  ~4 GB.
            lat_list.append(mu.reshape(b, T, *mu.shape[1:]).cpu().numpy().astype(np.float32))
            act_list.append(np.stack(acts))
            if i % 400 == 0:
                print(f"    {i}/{len(keep)}", flush=True)

    lat = np.concatenate(lat_list)
    act = np.concatenate(act_list).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, lat)
    np.save(cache_path.with_suffix(".actions.npy"), act)
    meta = {"n": int(lat.shape[0]), "frames": int(lat.shape[1]),
            "latent_channels": int(lat.shape[2]), "grid": int(lat.shape[-1]),
            "fraction": float(fraction), "split": split, "geometries": list(geometries),
            "seed": int(seed)}
    meta_path.write_text(json.dumps(meta))
    return np.load(cache_path, mmap_mode="r"), act, meta


# ==========================================================================
# Checkpointing
# ==========================================================================


class Trainer:
    """Training loop with wall-clock-budgeted, signal-safe checkpointing."""

    def __init__(self, outdir: Path, max_hours: float | None, keep_steps=()):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.deadline = None if not max_hours else time.time() + max_hours * 3600
        # Periodic checkpoints the ladder needs by name (checkpoint_at steps);
        # every other periodic checkpoint is pruned once a newer one lands.
        # ~1.1 GB each x 60 per run was ~70 GB/model of scratch for nothing.
        self.keep_steps = {int(s) for s in keep_steps}
        self._stop = False
        for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):  # not in main thread / unsupported
                pass

    def _on_signal(self, signum, frame):
        print(f"\n[trainer] signal {signum} received; checkpointing and exiting cleanly",
              flush=True)
        self._stop = True

    def should_stop(self) -> tuple[bool, str]:
        if self._stop:
            return True, "signal"
        if self.deadline and time.time() > self.deadline:
            return True, "wall-clock budget"
        return False, ""

    def latest(self) -> Path | None:
        cks = sorted(self.outdir.glob("ckpt_step*.pt"))
        return cks[-1] if cks else None

    def save(self, step: int, model, opt, sched, scaler, cfg, extra=None,
             tag: str | None = None) -> Path:
        """Atomic (tmp -> fsync -> rename); a half-written checkpoint on
        preemption would be worse than none."""
        path = self.outdir / (tag or f"ckpt_step{step:07d}.pt")
        tmp = path.with_suffix(".pt.tmp")
        payload = {
            "step": step,
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "cfg": cfg,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "extra": extra or {},
        }
        with open(tmp, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        if tag is None:
            self._prune(keep=path)
        return path

    def _prune(self, keep: Path) -> None:
        """Delete periodic checkpoints that are neither the newest nor a ladder
        checkpoint_at step.  Only after the newer one is fully on disk."""
        for p in sorted(self.outdir.glob("ckpt_step*.pt")):
            if p == keep:
                continue
            try:
                step = int(p.stem.replace("ckpt_step", ""))
            except ValueError:
                continue
            if step in self.keep_steps:
                continue
            try:
                p.unlink()
            except OSError as exc:  # never let housekeeping kill training
                print(f"  prune: could not remove {p.name}: {exc}", flush=True)

    def load(self, model, opt, sched, scaler) -> int:
        ck_path = self.latest()
        if ck_path is None:
            return 0
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        print(f"[trainer] resumed from {ck_path.name} at step {ck['step']}", flush=True)
        return int(ck["step"])


# ==========================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="WM-base")
    ap.add_argument("--vae", type=Path, default=C.CHECKPOINT_ROOT / "wm" / "vae" / "vae.pt")
    ap.add_argument("--clips", type=Path, default=C.DATA_ROOT / "wm_clips")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", type=Path, default=C.CHECKPOINT_ROOT / "wm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--geometries", nargs="+", default=list(C.GEOMETRIES))
    ap.add_argument("--max-hours", type=float,
                    default=float(os.environ.get("OGAF_MAX_HOURS", "0")) or None,
                    help="exit cleanly with a checkpoint after this much wall clock")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--budget-only", action="store_true",
                    help="print the FLOP/GPU-hour budget and exit without training")
    ap.add_argument("--cache-only", action="store_true",
                    help="build this model's latent cache and exit (a short pre-job, so the "
                         "first 2 h training window is spent training)")
    args = ap.parse_args()

    cfg = resolve_model(args.model)
    if args.steps:
        cfg["train"]["steps"] = args.steps
    t = cfg["transformer"]
    v = cfg["vae"]
    grid = C.RENDER_SIZE // v["downsample"]
    n_patch = (grid // t["patch"]) ** 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       ("mps" if torch.backends.mps.is_available() else "cpu"))

    model = VideoWorldModel(
        latent_channels=v["latent_channels"], grid=grid, patch=t["patch"],
        dim=t["dim"], depth=t["depth"], heads=t["heads"],
        max_frames=t["max_frames"],
    ).to(dev)

    budget = flop_budget(model, cfg, t["max_frames"], n_patch)
    print(f"=== {args.model} ===")
    print(f"  params            {budget['params_M']:.1f}M")
    print(f"  tokens/sample     {budget['tokens_per_sample']} "
          f"({t['max_frames']} frames x {n_patch} patches)")
    print(f"  TFLOP/step        {budget['tflop_per_step']:.2f} "
          f"(attention share {budget['attention_share']:.1%})")
    print(f"  steps             {budget['steps']}")
    print(f"  est. GPU-hours    A100 {budget['gpu_hours_a100']:.1f}  "
          f"L40S {budget['gpu_hours_l40s']:.1f}")
    if args.budget_only:
        print(json.dumps(budget, indent=2))
        return 0

    vck = torch.load(args.vae, map_location="cpu", weights_only=False)
    vae = CompactVAE(vck["latent_channels"]).to(dev).eval()
    vae.load_state_dict(vck["state_dict"])
    for p in vae.parameters():
        p.requires_grad_(False)

    fm = float(cfg["train"].get("friction_mult", 1.0))
    frac = float(cfg["train"].get("fraction", 1.0))
    clip_root = Path(args.clips)
    if abs(fm - 1.0) > 1e-9:
        clip_root = clip_root.parent / f"{clip_root.name}_fm{fm:g}"
    # Start the wall-clock budget NOW, before the latent cache is (re)built: a
    # first window that spends minutes encoding and only then starts its 1.8 h
    # budget would overrun the 2 h SLURM limit and be killed mid-checkpoint.
    outdir = Path(args.out) / args.model
    ladder_doc = yaml.safe_load(LADDER_PATH.read_text())
    keep_steps = [m["checkpoint_at"] for m in ladder_doc["models"] if "checkpoint_at" in m]
    trainer = Trainer(outdir, args.max_hours, keep_steps=keep_steps)

    cache = C.DATA_ROOT / "wm_latents" / f"{args.model}.npy"
    latents, actions, meta = build_latent_cache(
        vae, clip_root, args.geometries, cache, dev, frac, args.seed
    )
    print(f"  latents {latents.shape}  actions {actions.shape}", flush=True)
    if args.cache_only:
        print("  --cache-only: latent cache is built; exiting before training")
        return 0
    # TF32 for any fp32 matmul/conv left outside bf16 autocast (A100/L40S);
    # the fused AdamW kernel is a free ~10-20% on the optimiser step.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=0.01,
                            fused=(dev.type == "cuda"))
    steps = cfg["train"]["steps"]
    warm = cfg["train"]["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 *
        (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, steps - warm)))))
    # ladder.yaml registers precision: bf16.  autocast's default is fp16,
    # which would be a different (unregistered) precision and can overflow the
    # 18-sublayer residual stream; bf16 needs no loss scaling at all.
    use_amp = dev.type == "cuda"
    amp_dtype = (torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported())
                 else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and amp_dtype == torch.float16))
    print(f"  autocast: {'off' if not use_amp else str(amp_dtype).replace('torch.', '')}"
          f"  grad-scaler: {scaler.is_enabled()}", flush=True)

    start = trainer.load(model, opt, sched, scaler)
    if start >= steps:
        print(f"  already complete at step {start}")
        return 0

    bs = cfg["train"]["batch_size"]
    rng = np.random.default_rng(args.seed + start)
    hist, t0 = [], time.time()

    model.train()
    for step in range(start, steps):
        idx = np.sort(rng.integers(0, len(latents), size=bs))
        z = torch.from_numpy(np.asarray(latents[idx]).astype(np.float32)).to(dev)
        a = torch.from_numpy(actions[idx]).to(dev)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            pred = model(z[:, :-1], a)          # predict frames 1..T-1
            loss = F.mse_loss(pred, z[:, 1:])
        if not torch.isfinite(loss):
            # Stop rather than burn the allocation: a stalled run would print
            # NaN losses for hours and still "exit 0".
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % 200 == 0:
            hist.append({"step": step, "loss": float(loss)})
            el = time.time() - t0
            rate = (step - start + 1) / max(el, 1e-9)
            print(f"  step {step:7d}/{steps}  loss={float(loss):.6f}  "
                  f"{rate:.2f} it/s  eta {(steps - step) / max(rate, 1e-9) / 3600:.2f} h",
                  flush=True)

        stop, why = trainer.should_stop()
        if stop or (step > start and step % args.ckpt_every == 0):
            p = trainer.save(step, model, opt, sched, scaler, cfg)
            if stop:
                print(f"[trainer] stopped at step {step} ({why}); saved {p.name}.\n"
                      f"[trainer] re-run the SAME command to resume -- SLURM --requeue "
                      f"does exactly that.", flush=True)
                return 0

    trainer.save(steps, model, opt, sched, scaler, cfg, tag="final.pt")
    provenance.save_run(outdir, f"wm_train::{args.model}",
                        {"loss": np.asarray([h["loss"] for h in hist])},
                        seeds={"seed": args.seed},
                        extra={"config": cfg, "budget": budget, "latent_meta": meta,
                               "vae": str(args.vae)},
                        arrays_name="history.npz")
    print(f"wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
