"""Run provenance: `metadata.json` for every run (§0.3).

Every compute script writes, next to its `.npz` arrays, a `metadata.json`
containing the full config, the git SHA (and dirty flag), package versions
including mujoco and torch, and all seeds.  Summary statistics are *never*
computed here -- `analyze.py` does that, from the saved arrays.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import config as C

__all__ = ["git_info", "package_versions", "build_metadata", "save_run", "save_arrays"]

_TRACKED_PACKAGES = (
    "mujoco", "numpy", "torch", "torchvision", "transformers", "timm",
    "scipy", "sklearn", "statsmodels", "imageio", "PIL", "yaml",
)


def git_info(repo: Path | None = None) -> dict:
    repo = Path(repo or C.REPO_ROOT)

    def run(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True, text=True, timeout=15, check=True,
            ).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return f"<unavailable: {type(exc).__name__}>"

    status = run("status", "--porcelain")
    return {
        "sha": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) and not status.startswith("<unavailable"),
        "dirty_files": [line[3:] for line in status.splitlines()][:50] if status else [],
        "describe": run("describe", "--always", "--dirty", "--tags"),
    }


def package_versions() -> dict:
    # mj_env FIRST: it must reach os.environ before anything imports mujoco,
    # and the loop below imports mujoco to read its version (§14).
    import mj_env

    out = {"python": sys.version, "platform": platform.platform(),
           "mujoco_gl": mj_env.MUJOCO_GL}
    for name in _TRACKED_PACKAGES:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "<no __version__>")
        except Exception:  # noqa: BLE001
            out[name] = None
    try:
        import torch

        out["torch_cuda"] = torch.version.cuda
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:  # noqa: BLE001
        pass
    return out


def build_metadata(experiment: str, *, seeds=None, extra: dict | None = None) -> dict:
    return {
        "experiment": experiment,
        "created_unix": time.time(),
        "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git_info(),
        "packages": package_versions(),
        "config": C.as_dict(),
        "seeds": seeds,
        "argv": sys.argv,
        "extra": extra or {},
    }


def _jsonable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_run(outdir: Path, experiment: str, arrays: dict, *, seeds=None,
             extra: dict | None = None, arrays_name: str = "raw.npz") -> Path:
    """Persist raw per-sample arrays plus metadata.json (§0.3).

    No plotting, no summary statistics.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    import os

    # Atomic: a wall-clock kill mid-write must not leave a truncated eval.npz
    # that a skip-if-exists check then treats as a finished run.
    npz_path = save_arrays(outdir / arrays_name, **arrays)
    meta = build_metadata(experiment, seeds=seeds, extra=extra)
    meta["arrays"] = {k: list(np.shape(v)) for k, v in arrays.items()}
    meta["arrays_file"] = arrays_name
    mpath = outdir / "metadata.json"
    tmp = mpath.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(_jsonable(meta), fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, mpath)
    return npz_path


def save_arrays(path: Path, **arrays) -> Path:
    """Atomic `.npz` write (tmp -> fsync -> rename), for per-item corpus files."""
    import os

    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path
