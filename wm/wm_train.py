"""Action-conditioned latent video predictor for Experiment H (§9-H).

WM-base: a frozen compact VAE plus a ~75M-parameter spatiotemporal transformer
that predicts latent frames autoregressively, conditioned on ``a in R^3``
via AdaLN.

Any substitute architecture is acceptable provided it (a) is action-conditioned,
(b) trains on this corpus, and (c) passes the §13.3 S1 validators **including
the two-action divergence check** -- action conditioning being silently unwired
is the one bug that invalidates every downstream H number, and it is detectable
with two generations.

    python wm/wm_train.py --model WM-base
    python wm/wm_train.py --model WM-physics-corrupted
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C  # noqa: E402
import datasets as ds  # noqa: E402
import provenance  # noqa: E402
from wm.vae import CompactVAE  # noqa: E402

LADDER_PATH = Path(__file__).resolve().parent / "ladder.yaml"


# ==========================================================================
# Model
# ==========================================================================


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNBlock(nn.Module):
    """Transformer block with AdaLN action conditioning."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond):
        s1, sc1, g1, s2, sc2, g2 = self.ada(cond).chunk(6, dim=-1)
        h = modulate(self.n1(x), s1, sc1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * h
        h = modulate(self.n2(x), s2, sc2)
        x = x + g2.unsqueeze(1) * self.mlp(h)
        return x


class ActionConditionedPredictor(nn.Module):
    """Predicts the next latent frame from ``context_frames`` latents + action."""

    def __init__(self, latent_channels=4, grid=28, patch=2, dim=768, depth=12,
                 heads=12, context_frames=2, n_actions=3):
        super().__init__()
        self.grid, self.patch, self.ctx = grid, patch, context_frames
        self.latent_channels = latent_channels
        self.n_patch = (grid // patch) ** 2
        in_dim = latent_channels * patch * patch
        self.embed = nn.Linear(in_dim * context_frames, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.action_mlp = nn.Sequential(nn.Linear(n_actions, dim), nn.SiLU(),
                                        nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([AdaLNBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head = nn.Linear(dim, in_dim)
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _patchify(self, z):
        # z: (B, T, Cl, G, G) -> (B, n_patch, T*Cl*p*p)
        B, T, Cl, G, _ = z.shape
        p = self.patch
        z = z.reshape(B, T * Cl, G, G)
        z = z.unfold(2, p, p).unfold(3, p, p)             # (B, TC, G/p, G/p, p, p)
        z = z.permute(0, 2, 3, 1, 4, 5).reshape(B, self.n_patch, -1)
        return z

    def _unpatchify(self, x):
        B = x.shape[0]
        p, g = self.patch, self.grid // self.patch
        x = x.reshape(B, g, g, self.latent_channels, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, self.latent_channels, self.grid, self.grid)
        return x

    def forward(self, context, action):
        h = self.embed(self._patchify(context)) + self.pos
        cond = self.action_mlp(action)
        for blk in self.blocks:
            h = blk(h, cond)
        shift, scale = self.ada_out(cond).chunk(2, dim=-1)
        h = modulate(self.norm(h), shift, scale)
        return self._unpatchify(self.head(h))


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


def load_latents(vae: CompactVAE, geometries, friction_mult: float, fraction: float,
                 device, seed: int):
    """Encode the corpus's stored horizon frames into VAE latents.

    The stored frames are the four horizons, which is the sequence the world
    model is asked to predict: conditioning on (s_0, s_std) and predicting the
    settled frame is exactly the contact outcome under test.
    """
    data = ds.load_split("train", condition="INTERACT", friction_mult=friction_mult,
                         keys=["frames", "action", "tuple_index"])
    frames = data["frames"]          # (N, 4, H, W, 3)
    actions = data["action"]
    n = len(frames)
    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=max(1, int(n * fraction)), replace=False)
        frames, actions = frames[keep], actions[keep]
    lat = []
    with torch.no_grad():
        for i in range(0, len(frames), 32):
            x = torch.from_numpy(np.ascontiguousarray(frames[i : i + 32]))
            B, T = x.shape[:2]
            x = x.reshape(B * T, *x.shape[2:]).permute(0, 3, 1, 2).contiguous().float().to(device) / 255.0
            mu, _ = vae.encode(x)
            lat.append(mu.reshape(B, T, *mu.shape[1:]).cpu())
    return torch.cat(lat), torch.from_numpy(actions).float()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="WM-base")
    ap.add_argument("--vae", type=Path, default=C.CHECKPOINT_ROOT / "wm" / "vae" / "vae.pt")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", type=Path, default=C.CHECKPOINT_ROOT / "wm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=12000,
                    help="model 2 and 3 of the ladder are checkpoints of model 1 -- free")
    args = ap.parse_args()

    cfg = resolve_model(args.model)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vck = torch.load(args.vae, map_location="cpu", weights_only=False)
    vae = CompactVAE(vck["latent_channels"]).to(dev).eval()
    vae.load_state_dict(vck["state_dict"])
    for p in vae.parameters():
        p.requires_grad_(False)

    fm = float(cfg["train"].get("friction_mult", 1.0))
    frac = float(cfg["train"].get("fraction", 1.0))
    print(f"{args.model}: friction_mult={fm}, corpus fraction={frac}, device={dev}")
    latents, actions = load_latents(vae, C.GEOMETRIES, fm, frac, dev, args.seed)
    print(f"  encoded {len(latents)} rollouts -> latents {tuple(latents.shape)}")

    t = cfg["transformer"]
    grid = latents.shape[-1]
    model = ActionConditionedPredictor(
        latent_channels=latents.shape[2], grid=grid, patch=t["patch"], dim=t["dim"],
        depth=t["depth"], heads=t["heads"], context_frames=t["context_frames"],
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  predictor: {n_params / 1e6:.1f}M params")

    steps = args.steps or cfg["train"]["steps"]
    bs = cfg["train"]["batch_size"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=0.01)
    warm = cfg["train"]["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) *
        0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, steps - warm)))))

    outdir = Path(args.out) / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    hist = []
    ctx = t["context_frames"]

    model.train()
    for step in range(steps):
        idx = rng.integers(0, len(latents), size=bs)
        z = latents[idx].to(dev)                    # (B, 4, Cl, G, G)
        a = actions[idx].to(dev)
        # Condition on (s_0, s_std); predict s_del -- the contact outcome.
        context = z[:, :ctx]
        target = z[:, ctx]
        pred = model(context, a)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            hist.append({"step": step, "loss": float(loss)})
            print(f"  step {step:6d}  loss={float(loss):.6f}", flush=True)
        if args.checkpoint_every and step > 0 and step % args.checkpoint_every == 0:
            torch.save({"state_dict": model.state_dict(), "step": step,
                        "cfg": cfg, "vae": str(args.vae)},
                       outdir / f"step{step:06d}.pt")

    torch.save({"state_dict": model.state_dict(), "step": steps, "cfg": cfg,
                "vae": str(args.vae)}, outdir / "final.pt")
    provenance.save_run(outdir, f"wm_train::{args.model}",
                        {"loss": np.asarray([h["loss"] for h in hist])},
                        seeds={"seed": args.seed},
                        extra={"config": cfg, "n_params": n_params, "steps": steps,
                               "friction_mult": fm, "corpus_fraction": frac},
                        arrays_name="history.npz")
    print(f"wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
