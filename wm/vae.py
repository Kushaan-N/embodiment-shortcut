"""Compact VAE for Experiment H (§9-H).

A small convolutional VAE trained on this project's own corpus frames, used
frozen as the latent space for the action-conditioned video predictor.

Trained here rather than taken off the shelf for one reason: whatever encodes
these frames must preserve contact-scale geometry, and Gate B has already
measured what "contact-scale" means (``delta_pos_min``).  ``verify_fidelity``
checks that a ``delta_pos_min`` object displacement survives an
encode/decode round-trip.  An off-the-shelf sd-vae is acceptable in its place
**only if** it passes the same check (§9-H).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C  # noqa: E402
import provenance  # noqa: E402


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(8, ch), nn.GroupNorm(8, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class CompactVAE(nn.Module):
    """224 -> 28 latent grid, ``latent_channels`` deep (downsample 8)."""

    def __init__(self, latent_channels: int = 4, base: int = 64, downsample: int = 8):
        super().__init__()
        n_down = int(np.log2(downsample))
        chs = [base * min(2 ** i, 4) for i in range(n_down)]
        enc, cin = [nn.Conv2d(3, chs[0], 3, padding=1)], chs[0]
        for ch in chs:
            enc += [nn.Conv2d(cin, ch, 4, stride=2, padding=1), ResBlock(ch)]
            cin = ch
        enc += [nn.GroupNorm(8, cin), nn.SiLU(), nn.Conv2d(cin, 2 * latent_channels, 1)]
        self.encoder = nn.Sequential(*enc)

        dec, cin = [nn.Conv2d(latent_channels, chs[-1], 3, padding=1)], chs[-1]
        for ch in reversed(chs):
            dec += [ResBlock(cin), nn.ConvTranspose2d(cin, ch, 4, stride=2, padding=1)]
            cin = ch
        dec += [nn.GroupNorm(8, cin), nn.SiLU(), nn.Conv2d(cin, 3, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)
        self.latent_channels = latent_channels
        self.downsample = downsample

    def encode(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar.clamp(-20, 10)

    def reparam(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar) if self.training else mu
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar, kl_weight: float = 1e-6):
    rec = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + kl_weight * kl, {"recon": float(rec), "kl": float(kl)}


@torch.no_grad()
def verify_fidelity(model: CompactVAE, geometry: str = "box",
                    device=None, n_dirs: int = 8) -> dict:
    """Does a ``delta_pos_min`` object displacement survive the round-trip?

    The same question Gate B asks of DINO, asked of the VAE.  A latent space
    that cannot represent the smallest physically meaningful motion cannot
    support a world model whose contact outcomes are being scored.
    """
    import scene

    device = device or next(model.parameters()).device
    thr = C.load_thresholds().for_geometry(geometry)
    d_min = thr["delta_pos_min"]
    arm = np.array([*C.REST_POSE, C.REST_YAW])
    h = scene.object_rest_height(geometry)
    base = np.array([0.055, 0.015, h, 1.0, 0.0, 0.0, 0.0])

    def embed(pose):
        img = scene.render_state(geometry, "INTERACT", arm, pose)
        x = torch.from_numpy(img).permute(2, 0, 1)[None].contiguous().float().to(device) / 255.0
        mu, _ = model.encode(x)
        return mu.flatten().cpu().numpy(), x

    z0, x0 = embed(base)
    rng = np.random.default_rng(0)
    shifted, recon_err = [], []
    for _ in range(n_dirs):
        d = rng.normal(size=2)
        d /= np.linalg.norm(d)
        p = base.copy()
        p[0] += d_min * d[0]
        p[1] += d_min * d[1]
        z1, _ = embed(p)
        shifted.append(float(np.linalg.norm(z1 - z0)))
    # Noise floor: identical render twice (deterministic -> exactly 0; the
    # meaningful floor is the reconstruction error scale instead).
    rec, _, _ = model(x0)
    recon_err.append(float(F.mse_loss(rec, x0)))

    return {
        "delta_pos_min_m": d_min,
        "latent_shift_at_delta_min_mean": float(np.mean(shifted)),
        "latent_shift_at_delta_min_min": float(np.min(shifted)),
        "latent_norm": float(np.linalg.norm(z0)),
        "relative_shift": float(np.mean(shifted) / max(np.linalg.norm(z0), 1e-9)),
        "recon_mse": float(np.mean(recon_err)),
        "note": "a delta_pos_min displacement must move the latent measurably; "
                "compare against Gate B's finding for the same displacement",
    }


def main() -> int:
    import datasets as ds

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--latent-channels", type=int, default=4)
    ap.add_argument("--geometries", nargs="+", default=list(C.GEOMETRIES))
    ap.add_argument("--out", type=Path, default=C.CHECKPOINT_ROOT / "wm" / "vae")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ds.load_split("train", geometry=None, condition="INTERACT",
                         keys=["frames"])
    frames = data["frames"].reshape(-1, C.RENDER_SIZE, C.RENDER_SIZE, 3)
    print(f"VAE training on {len(frames)} frames, device={dev}")

    model = CompactVAE(args.latent_channels).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    rng = np.random.default_rng(args.seed)
    hist = []

    model.train()
    for step in range(args.steps):
        idx = rng.integers(0, len(frames), size=args.batch_size)
        x = torch.from_numpy(np.ascontiguousarray(frames[idx]))
        x = x.permute(0, 3, 1, 2).contiguous().float().to(dev) / 255.0
        recon, mu, logvar = model(x)
        loss, parts = vae_loss(recon, x, mu, logvar)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            hist.append({"step": step, **parts})
            print(f"  step {step:6d}  recon={parts['recon']:.6f}  kl={parts['kl']:.4f}",
                  flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "latent_channels": args.latent_channels}, args.out / "vae.pt")
    model.eval()
    fidelity = verify_fidelity(model, device=dev)
    print("\nfidelity check:", fidelity)
    provenance.save_run(args.out, "wm_vae", {"history_recon":
                                             np.asarray([h["recon"] for h in hist])},
                        seeds={"seed": args.seed},
                        extra={"fidelity": fidelity, "steps": args.steps},
                        arrays_name="history.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
