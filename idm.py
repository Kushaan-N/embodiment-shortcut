"""Inverse dynamics models: Architecture A and Architecture B (§8).

Substrate-agnostic on purpose: pure PyTorch, paths from environment variables,
**no Modal imports anywhere in this file**, so the identical script runs on
Modal, on Unity, and locally.

Architecture A (§8.1, T1)
    The faithful reproduction of the published protocol.  Every
    faithfulness-critical hyperparameter comes from ``protocol.py``, which
    refuses to supply a value that has no recorded provenance.  The C1 claim
    must hold for *this* model -- a small MLP on frozen features would be the
    strawman T1 forbids.

Architecture B (§8.1)
    Frozen DINO + a small MLP.  Cheap, interpretable, 10 seeds.  Reported
    alongside, never the basis of a claim.

Masking decomposition (§8.4)
    Architecture B applied to full-frame / arm-masked / object-masked inputs.
    Masked regions are filled with the *background render*, not black, so the
    model cannot key on mask shape.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C

__all__ = [
    "TrainConfig", "ArchitectureA", "ArchitectureB", "build_model",
    "train_idm", "evaluate_idm", "normalise_action", "denormalise_action",
    "IMAGENET_MEAN", "IMAGENET_STD", "AppearanceAugment", "composite_mask",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==========================================================================
# Action normalisation
# ==========================================================================


def normalise_action(a: np.ndarray) -> np.ndarray:
    """Map actions to [0, 1] per dimension, using the sampling ranges (§7.1).

    Errors are reported in these units throughout, which is what makes the
    prior baseline exactly 0.25 MAE / 1/12 MSE for every dimension (§7.3).
    """
    lo, hi = C.action_ranges_array()
    return (np.asarray(a, dtype=np.float64) - lo) / (hi - lo)


def denormalise_action(x: np.ndarray) -> np.ndarray:
    lo, hi = C.action_ranges_array()
    return np.asarray(x, dtype=np.float64) * (hi - lo) + lo


# ==========================================================================
# Appearance augmentation (§8.3)
# ==========================================================================


class AppearanceAugment(nn.Module):
    """Mild colour jitter / blur / JPEG-like artefacts, applied on GPU.

    Used only for the IDM copies that score *generated* video in Experiment H,
    to shrink the sim->generated domain gap.  Clean copies everywhere else, and
    H reports both.
    """

    def __init__(self, strength: float = 1.0, enabled: bool = True):
        super().__init__()
        self.strength = float(strength)
        self.enabled = bool(enabled)

    def forward(self, x: torch.Tensor, generator: torch.Generator | None = None):
        # x: (B, ..., 3, H, W) float in [0, 1]
        if not self.enabled or not self.training:
            return x
        s = self.strength
        shape = list(x.shape)
        lead = shape[:-3]
        dev = x.device

        def r(lo, hi):
            return torch.empty(lead, device=dev).uniform_(lo, hi).reshape(*lead, 1, 1, 1)

        x = x * r(1 - 0.25 * s, 1 + 0.25 * s)                      # brightness
        mean = x.mean(dim=(-3, -2, -1), keepdim=True)
        x = mean + (x - mean) * r(1 - 0.3 * s, 1 + 0.3 * s)        # contrast
        grey = x.mean(dim=-3, keepdim=True)
        x = grey + (x - grey) * r(1 - 0.3 * s, 1 + 0.3 * s)        # saturation
        x = x + torch.randn_like(x) * (0.02 * s)                   # sensor noise

        # Gaussian blur, separable, fixed 5-tap kernel with random sigma
        flat = x.reshape(-1, *shape[-3:])
        sigma = float(torch.empty(1).uniform_(0.1, 1.0 * s + 0.1))
        k = torch.arange(-2, 3, device=dev, dtype=x.dtype)
        w = torch.exp(-(k ** 2) / (2 * sigma ** 2))
        w = (w / w.sum()).reshape(1, 1, 1, 5).repeat(3, 1, 1, 1)
        flat = F.conv2d(F.pad(flat, (2, 2, 0, 0), mode="reflect"), w, groups=3)
        flat = F.conv2d(F.pad(flat, (0, 0, 2, 2), mode="reflect"),
                        w.transpose(-1, -2), groups=3)
        # JPEG-like 8x8 blockiness, cheaply approximated by quantising blocks
        levels = 32.0
        flat = torch.round(flat * levels) / levels
        return flat.reshape(*shape).clamp(0, 1)


def composite_mask(frames: np.ndarray, masks: np.ndarray, background: np.ndarray,
                   keep: int) -> np.ndarray:
    """Keep only ``keep``-labelled pixels; fill the rest from the background render.

    Filling with the empty-scene render rather than black is what stops the
    probe from reading the *shape of the mask* instead of the content it was
    supposed to isolate (§8.4).

    ``masks`` may be given with or without a trailing channel axis.  Boolean
    fancy-indexing does NOT broadcast a trailing size-1 axis -- it raises --
    so the selector is explicitly broadcast to the frame shape here rather
    than relying on numpy to do it.
    """
    frames = np.asarray(frames)
    masks = np.asarray(masks)
    background = np.asarray(background)
    if masks.ndim == frames.ndim - 1:
        masks = masks[..., None]
    if masks.ndim != frames.ndim:
        raise ValueError(
            f"mask ndim {masks.ndim} incompatible with frames ndim {frames.ndim}"
        )
    out = np.broadcast_to(background, frames.shape).copy()
    sel = np.broadcast_to(masks == keep, frames.shape)
    out[sel] = frames[sel]
    return out


# ==========================================================================
# Architecture A -- faithful reproduction (T1)
# ==========================================================================


@dataclass
class TrainConfig:
    variant: str
    seed: int
    epochs: int
    lr: float
    batch_size: int
    weight_decay: float
    optimizer: str
    lr_schedule: str
    augment: bool = False
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    num_workers: int = 4
    amp: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class ArchitectureA(nn.Module):
    """ResNet-50 backbone + non-causal temporal aggregation, end-to-end.

    Consumes a clip ``(B, T, 3, H, W)``.  Pair variants (A-del, A-time) pass
    ``T = 2`` -- the same module, so the *only* thing that differs between the
    standard metric and OG-AF is which frames go in (§8.1), which is exactly
    the comparison the paper needs.
    """

    def __init__(self, proto, n_actions: int = 3):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        if proto["backbone"] != "resnet50":
            raise ValueError(f"protocol requires backbone {proto['backbone']}")
        weights = (ResNet50_Weights.IMAGENET1K_V2
                   if proto["backbone_init"] == "imagenet_pretrained" else None)
        net = resnet50(weights=weights)
        self.trunk = nn.Sequential(*list(net.children())[:-1])   # -> (B, 2048, 1, 1)
        d = int(proto["temporal_width"])
        self.proj = nn.Linear(2048, d)
        self.pos = nn.Parameter(torch.zeros(1, 512, d))
        nn.init.trunc_normal_(self.pos, std=0.02)

        if proto["temporal_aggregation"] != "noncausal_transformer":
            raise ValueError(f"unsupported temporal_aggregation {proto['temporal_aggregation']}")
        # VPT's IDM begins with a non-causal temporal convolution, then applies
        # residual transformer layers that are NOT causally masked.
        self.temporal_conv = nn.Conv1d(d, d, kernel_size=3, padding=1)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=8, dim_feedforward=4 * d, dropout=0.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=int(proto["temporal_layers"]))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, n_actions))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        B, T = clip.shape[:2]
        # .contiguous() is load-bearing: the loaders hand over permuted (and so
        # non-contiguous) uint8->float views, and reshape on those produces a
        # graph whose backward pass raises at runtime.
        x = clip.contiguous().reshape(B * T, *clip.shape[2:])
        x = (x - self.mean) / self.std
        f = self.trunk(x).flatten(1)              # (B*T, 2048)
        f = self.proj(f).reshape(B, T, -1)
        f = f + self.pos[:, :T]
        f = f + self.temporal_conv(f.transpose(1, 2)).transpose(1, 2)
        f = self.temporal(f)                      # non-causal: no mask passed
        return self.head(f.mean(dim=1))           # one action per clip


# ==========================================================================
# Architecture B -- frozen DINO + MLP
# ==========================================================================


class ArchitectureB(nn.Module):
    """MLP on pre-computed frozen-encoder embeddings.

    Takes ``[embed(s_0), embed(s_h)]`` already concatenated, so the expensive
    encoder pass happens once per frame for all 10 seeds.
    """

    def __init__(self, d_in: int, n_actions: int = 3, width: int = 512, depth: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)]
            d = width
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(variant: str, proto=None, *, d_in: int | None = None) -> nn.Module:
    if variant.startswith("A-"):
        if proto is None:
            raise ValueError("Architecture A requires the protocol record (§0.6)")
        return ArchitectureA(proto)
    if variant.startswith("B-"):
        if d_in is None:
            raise ValueError("Architecture B requires the embedding dimension")
        return ArchitectureB(d_in)
    raise ValueError(f"unknown variant {variant!r}")


# ==========================================================================
# Training / evaluation
# ==========================================================================


def _make_optimizer(model: nn.Module, cfg: TrainConfig):
    if cfg.optimizer != "adamw":
        raise ValueError(f"unsupported optimizer {cfg.optimizer!r}")
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _make_schedule(opt, cfg: TrainConfig, total_steps: int):
    if cfg.lr_schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    if cfg.lr_schedule != "cosine_with_warmup":
        raise ValueError(f"unsupported lr_schedule {cfg.lr_schedule!r}")
    warmup = max(1, int(cfg.warmup_frac * total_steps))

    def fn(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def train_idm(model: nn.Module, train_loader, val_loader, cfg: TrainConfig,
              *, dev=None, log_every: int = 50, augment: AppearanceAugment | None = None) -> dict:
    """Train one IDM.  Loss is MSE, because MultiWorld §B.2 says MSE.

    Model selection is on validation MAE (§7.1 makes MAE primary for
    *reporting*; the training objective stays the paper's).  Returns the
    training history; per-sample errors come from ``evaluate_idm``.
    """
    dev = dev or device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = model.to(dev)
    opt = _make_optimizer(model, cfg)
    total_steps = cfg.epochs * max(1, len(train_loader))
    sched = _make_schedule(opt, cfg, total_steps)
    use_amp = bool(cfg.amp and dev.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_mse": [], "val_mse": [], "val_mae": [], "lr": []}
    best = {"val_mae": float("inf"), "epoch": -1, "state": None}

    for epoch in range(cfg.epochs):
        model.train()
        running, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if augment is not None:
                x = augment(x)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = F.mse_loss(model(x), y)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(loss.detach())
            nb += 1

        val = evaluate_idm(model, val_loader, dev=dev)
        history["train_mse"].append(running / max(nb, 1))
        history["val_mse"].append(val["mse"])
        history["val_mae"].append(val["mae"])
        history["lr"].append(float(sched.get_last_lr()[0]))
        if val["mae"] < best["val_mae"]:
            best = {"val_mae": val["mae"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1:
            print(f"    epoch {epoch:3d}/{cfg.epochs}  train_mse={history['train_mse'][-1]:.5f}  "
                  f"val_mae={val['mae']:.5f}  lr={history['lr'][-1]:.2e}", flush=True)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    history["best_epoch"] = best["epoch"]
    history["best_val_mae"] = best["val_mae"]
    return history


@torch.no_grad()
def evaluate_idm(model: nn.Module, loader, *, dev=None, return_per_sample: bool = False) -> dict:
    """Evaluate; optionally return raw per-sample error arrays (§0.3).

    Errors are in normalised action units, so they are directly comparable to
    the §7.3 prior baseline of 0.25 (MAE) / 1/12 (MSE).
    """
    dev = dev or device()
    model = model.to(dev).eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(dev, non_blocking=True)
        preds.append(model(x).float().cpu().numpy())
        targets.append(y.numpy())
    if not preds:
        raise RuntimeError("empty evaluation loader")
    P = np.concatenate(preds)
    Y = np.concatenate(targets)
    abs_err = np.abs(P - Y)
    sq_err = (P - Y) ** 2
    out = {
        "mae": float(abs_err.mean()),
        "mse": float(sq_err.mean()),
        "mae_per_dim": abs_err.mean(0).tolist(),
        "mse_per_dim": sq_err.mean(0).tolist(),
        "n": int(len(Y)),
    }
    if return_per_sample:
        out["abs_err"] = abs_err
        out["sq_err"] = sq_err
        out["pred"] = P
        out["target"] = Y
    return out


def train_config_from_protocol(proto, variant: str, seed: int, *, augment: bool = False,
                               epochs: int | None = None) -> TrainConfig:
    """Build a TrainConfig whose every field is protocol-sourced (§0.6)."""
    return TrainConfig(
        variant=variant,
        seed=seed,
        epochs=int(epochs if epochs is not None else proto["epochs"]),
        lr=float(proto["learning_rate"]),
        batch_size=int(proto["batch_size"]),
        weight_decay=float(proto["weight_decay"]),
        optimizer=str(proto["optimizer"]),
        lr_schedule=str(proto["lr_schedule"]),
        augment=augment,
    )
