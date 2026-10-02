"""Regression tests for the Unity-era plumbing: per-epoch resume, ladder
checkpoint resolution, the paired C3(b) effect, and the registered threshold
parse.  CPU-only and small; no corpus or GPU needed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import idm  # noqa: E402
import idm_data as idd  # noqa: E402


def _tiny_problem(seed=0, n=256, d=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    W = rng.normal(size=(d, 3)).astype(np.float32)
    Y = np.tanh(X @ W) * 0.5
    return X, Y


def _cfg(epochs, seed=0):
    return idm.TrainConfig(variant="B-std", seed=seed, epochs=epochs, lr=1e-2, batch_size=32,
                           weight_decay=0.0, optimizer="adamw", lr_schedule="cosine_with_warmup",
                           augment=False, num_workers=0, amp=False)


def _run(epochs, resume_path, seed=0):
    X, Y = _tiny_problem()
    torch.manual_seed(seed)
    model = idm.build_model("B-std", d_in=X.shape[1])
    tr = idd.make_loader(idd.EmbeddingDataset(X[:200], Y[:200], {}), 32, shuffle=True, seed=seed)
    va = idd.make_loader(idd.EmbeddingDataset(X[200:], Y[200:], {}), 32, shuffle=False)
    hist = idm.train_idm(model, tr, va, _cfg(epochs, seed), dev=torch.device("cpu"),
                         resume_path=resume_path)
    return model, hist


def test_resume_continues_from_the_next_epoch_and_matches_history_length(tmp_path):
    rp = tmp_path / "resume.pt"
    # Interrupted run: 2 epochs of a 4-epoch config.  Simulated by running a
    # 2-epoch config to completion with the same resume file.
    _, h2 = _run(epochs=2, resume_path=rp)
    assert rp.exists() and len(h2["val_mae"]) == 2
    ck = torch.load(rp, map_location="cpu", weights_only=False)
    assert ck["epoch"] == 1 and set(ck) >= {"model", "optimizer", "scheduler", "history", "best",
                                            "torch_rng", "numpy_rng", "loader_rng"}
    # Relaunch with the full 4-epoch config: must start at epoch 2, not 0.
    model, h4 = _run(epochs=4, resume_path=rp)
    assert len(h4["val_mae"]) == 4, "history was not carried across the resume"
    assert h4["val_mae"][:2] == h2["val_mae"][:2]
    assert h4["best_epoch"] >= 0 and np.isfinite(h4["best_val_mae"])


def test_resume_file_is_optional_and_absent_means_fresh_start(tmp_path):
    _, h = _run(epochs=1, resume_path=tmp_path / "never_written_before.pt")
    assert len(h["val_mae"]) == 1


def test_resolve_checkpoint_follows_the_ladder(tmp_path):
    from wm.wm_generate import resolve_checkpoint
    import yaml
    from wm.wm_generate import LADDER_PATH

    ladder = yaml.safe_load(LADDER_PATH.read_text())
    base = ladder["base"]["name"]
    steps = int(ladder["base"]["train"]["steps"])
    root = tmp_path / "wm"
    (root / base).mkdir(parents=True)
    for m in ladder["models"]:
        if "checkpoint_at" in m and int(m["checkpoint_at"]) < steps:
            (root / base / f"ckpt_step{int(m['checkpoint_at']):07d}.pt").write_bytes(b"x")
    (root / base / "final.pt").write_bytes(b"x")
    for m in ladder["models"]:
        p = resolve_checkpoint(m["name"], root)
        if "checkpoint_at" in m:
            assert p.parent.name == base
            if int(m["checkpoint_at"]) >= steps:
                assert p.name == "final.pt"
            else:
                assert p.name == f"ckpt_step{int(m['checkpoint_at']):07d}.pt"
        else:
            assert p == root / m["name"] / "final.pt"
    # A pruned/missing ladder checkpoint must be a loud error, not a silent fallback.
    (root / base / "ckpt_step0012000.pt").unlink()
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("WM-base-10", root)


def test_paired_effect_uses_the_pairing():
    from wm.exp_h import cohens_d, paired_effect

    rng = np.random.default_rng(0)
    t = np.arange(200)
    a = rng.normal(0, 1, 200)
    b = a + 0.3 + rng.normal(0, 0.1, 200)     # same per-tuple component + a shift
    p = paired_effect(a, t, b, t, n_boot=500)
    assert p["n_pairs"] == 200 and p["unmatched"] == 0
    assert abs(p["mean_diff"] - 0.3) < 0.03
    assert p["ci_low"] > 1.5 > cohens_d(b, a)   # paired >> unpaired on shared variance
    q = paired_effect(a, t, b, t[::-1], n_boot=200)  # misaligned ids: the effect collapses
    assert abs(q["d_paired"]) < 0.5
    assert paired_effect(a[:2], t[:2], b[:2], t[:2])["n_pairs"] == 2   # degenerate, no crash


def test_registered_thresholds_parse_from_prereg_md():
    import analyze
    import prereg_lock

    if not prereg_lock.PREREG_PATH.exists():
        pytest.skip("no prereg.md in this checkout")
    got = analyze._read_prereg_thresholds()
    assert got.get("tau_1") == 0.25 and got.get("eps_a") == 0.10


def test_generation_validator_accepts_a_correct_still_start():
    """The real clip is still during the hold phase; a correct generation is too."""
    from wm.wm_generate import validate_item

    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(224, 224, 3)).astype(np.uint8)
    frames = np.stack([base, base] + [np.clip(base.astype(int) + rng.integers(-6, 7, base.shape), 0, 255).astype(np.uint8)
                                      for _ in range(14)])
    chk = validate_item(frames, base, expected_count=16)
    assert chk["_all_passed"], {k: v for k, v in chk.items() if isinstance(v, dict) and not v["passed"]}
    still = np.stack([base] * 16)
    assert not validate_item(still, base, expected_count=16)["_all_passed"]
