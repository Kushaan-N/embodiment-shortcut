"""Unity preflight (§13.4).  One line per check; exit non-zero on any failure.

Run this on a *compute* node, not a login node -- several checks are meaningless
elsewhere, especially the EGL render.

The check that earns its keep: **an actual one-frame MuJoCo EGL render**.
Checking that `MUJOCO_GL=egl` is set is not enough -- headless GL silently falls
back to osmesa, which works but is ~10x slower, and would corrupt the S2
wall-clock estimate and therefore the S4 `--time` request.

    srun -p gpu --gres=gpu:1 --time=00:15:00 python unity/preflight.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = []


def check(name: str, fn, *, required: bool = True):
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    RESULTS.append({"check": name, "passed": bool(ok), "required": required,
                    "detail": detail})
    flag = "PASS" if ok else ("FAIL" if required else "warn")
    print(f"[{flag:4s}] {name:44s} {detail}")
    return ok


# ---------------------------------------------------------------- checks


def c_dataset_mirrors():
    paths = [Path("/datasets/ai/dinov2"), Path("/datasets/ai/nvidia")]
    found = {str(p): p.exists() for p in paths}
    return any(found.values()), json.dumps(found)


def c_weight_checksums(manifest: Path):
    """A truncated download looks complete; only a checksum catches it."""
    if not manifest.exists():
        return False, f"{manifest} missing (write one listing sha256 per weight file)"
    import hashlib

    entries = json.loads(manifest.read_text())
    bad = []
    for rel, expect in entries.items():
        p = Path(rel)
        if not p.exists():
            bad.append(f"{rel}: missing")
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expect:
            bad.append(f"{rel}: checksum mismatch")
    return not bad, ("all match" if not bad else "; ".join(bad[:5]))


def c_hf_token():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        return False, "HF_TOKEN unset (needed for gated encoders, e.g. DINOv3)"
    return len(tok) > 20, f"present, length {len(tok)}"


def c_encoder_cache():
    """The Gate B encoder must resolve from cache.

    Every sbatch script runs with HF_HUB_OFFLINE=1, so a cache miss is not a
    slow download -- it is a hard failure at model load, after the job has
    queued and been allocated a GPU.  Unity's /datasets/ai/dinov2 mirror does
    NOT satisfy this: DINOv2 was measured and rejected at 0.55 sigma against a
    3 sigma threshold (config.GATE_B_EVIDENCE).
    """
    import config as C

    repo = C.GATE_B_SELECTED_ENCODER
    where = os.environ.get("HF_HOME") or "(HF_HOME unset -> $HOME/.cache)"
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False, "huggingface_hub not importable"

    def cached(fname):
        return isinstance(try_to_load_from_cache(repo, fname), str)

    if not cached("config.json"):
        return False, f"{repo} not cached under HF_HOME={where}; pre-stage it"
    weights = [f for f in ("model.safetensors", "pytorch_model.bin")
               if cached(f)]
    if not weights:
        return False, (f"{repo}: config.json cached but no weight file under "
                       f"HF_HOME={where} (a config-only cache still fails offline)")
    return True, f"{repo} [{weights[0]}] cached under HF_HOME={where}"


def c_thresholds():
    """Every script calls config.load_thresholds(); on Unity OGAF_RESULTS is the
    workspace, so the committed thresholds.json is only found via
    OGAF_THRESHOLDS (unity/env.sh sets it).  Never re-derive on Unity."""
    import config as C

    p = C.THRESHOLDS_PATH
    if not p.exists():
        return False, (f"{p} missing; `source unity/env.sh` pins OGAF_THRESHOLDS to the "
                       f"committed results/exp_a/thresholds.json -- do NOT re-run exp_a")
    return True, str(p)


def c_resnet50_weights():
    """Architecture A builds resnet50(weights=IMAGENET1K_V2); torchvision
    downloads it on first use, and compute nodes have no network -- so every
    Arch A task died at build_model after its GPU was allocated."""
    home = os.environ.get("TORCH_HOME")
    if not home:
        return False, "TORCH_HOME unset; `source unity/env.sh`"
    p = Path(home) / "hub" / "checkpoints" / "resnet50-11ad3fa6.pth"
    if not p.exists():
        return False, (f"{p} missing; pre-stage on a LOGIN node with env.sh sourced: "
                       "python -c 'from torchvision.models import resnet50, ResNet50_Weights; "
                       "resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)'")
    mb = p.stat().st_size / 2**20
    return mb > 90, f"{p} ({mb:.0f} MiB)"


def c_frame_stores():
    """train_idm.py opens the memmapped frame stores; they are built by
    `python idm_data.py` (CPU) after the corpus.  Warn-only: preflight also runs
    before any corpus exists."""
    import config as C

    root = C.DATA_ROOT / "frame_store"
    have = sorted(p.name for p in root.glob("*.index.json")) if root.exists() else []
    need = len(C.GEOMETRIES) * len(C.CONDITIONS)
    ok = len(have) >= need
    return ok, (f"{len(have)}/{need} index files under {root}"
                + ("" if ok else "; run `python idm_data.py` after the corpus build"))


def c_caches_off_home():
    """Caches under $HOME will fill a shared group quota (see unity/env.sh)."""
    home = str(Path.home())
    offenders = {v: os.environ.get(v) for v in
                 ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "PIP_CACHE_DIR",
                  "UV_CACHE_DIR", "TRITON_CACHE_DIR")
                 if not os.environ.get(v) or os.environ.get(v, "").startswith(home)}
    if offenders:
        return False, (f"unset or under $HOME: {sorted(offenders)}; "
                       f"`source unity/env.sh` first")
    return True, "all caches point off $HOME"


def c_cuda():
    import torch

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    i = torch.cuda.get_device_properties(0)
    return True, (f"{i.name}, {i.total_memory / 2**30:.1f} GiB, "
                  f"torch.version.cuda={torch.version.cuda}")


def c_egl_render():
    """The check that actually matters (§13.4)."""
    import mj_env

    # Warm up once: the timed region includes EGL context creation, and a cold
    # context on a healthy node can exceed the 0.5 s osmesa heuristic.  Time
    # the second render.
    mj_env.assert_real_render()
    info = mj_env.assert_real_render()
    slow = info["render_seconds"] > 0.5
    return (not slow), (f"backend={info['mujoco_gl_env']} "
                        f"{info['render_seconds'] * 1e3:.0f} ms/frame "
                        f"std={info['pixel_std']:.1f}"
                        + ("  <-- SLOW: likely osmesa fallback" if slow else ""))


def c_offline_ok():
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    r = subprocess.run([sys.executable, "-c",
                        "import transformers, torch; print('ok')"],
                       capture_output=True, text=True, env=env, timeout=180)
    return r.returncode == 0, (r.stdout or r.stderr).strip()[:120]


def c_squota(projected_gb: float):
    if not shutil.which("squota"):
        return False, "squota not found (are you on Unity?)"
    r = subprocess.run(["squota"], capture_output=True, text=True, timeout=60)
    return r.returncode == 0, (f"needs >= {3 * projected_gb:.0f} GB free "
                               f"(3x projected); output: {r.stdout.strip()[:160]}")


def c_workspace(projected_days: float):
    if not shutil.which("ws_list"):
        return False, "ws_list not found"
    r = subprocess.run(["ws_list", "-v"], capture_output=True, text=True, timeout=60)
    txt = r.stdout
    note = (f"needs >= {2 * projected_days:.0f} remaining days (2x projected) AND "
            f">= 1 extension left; workspaces are 30 days max, NO snapshots, NO recovery")
    return r.returncode == 0 and bool(txt.strip()), f"{note}; {txt.strip()[:160]}"


def c_home_quota():
    r = subprocess.run(["df", "-h", str(Path.home())], capture_output=True, text=True,
                       timeout=30)
    return r.returncode == 0, r.stdout.strip().splitlines()[-1][:120]


def c_time_and_partition(time_str: str, partition: str, qos: str | None):
    """The real trap: the default time limit is 1 HOUR on every partition."""
    if not time_str:
        return False, "--time is MANDATORY; without it jobs die at exactly 60 minutes"
    parts = partition.split(",")
    notes = [f"--time={time_str}", f"-p {partition}"]
    if qos:
        notes.append(f"--qos={qos}")
    hours = _time_to_hours(time_str)
    if hours > 48 and qos != "long":
        return False, f"--time={time_str} (>2 days) requires --qos=long"
    if "gpu-preempt" in parts and hours > 2:
        return False, ("gpu-preempt jobs can be killed after 2 hours; no work item "
                       "may exceed 2h")
    if hours > 14 * 24:
        return False, "gpu / gpu-preempt allow at most 14 days"
    return True, "; ".join(notes)


def _time_to_hours(t: str) -> float:
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
    bits = [float(x) for x in t.split(":")]
    while len(bits) < 3:
        bits.append(0.0)
    return days * 24 + bits[0] + bits[1] / 60 + bits[2] / 3600


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time", default="", help="the --time you intend to request")
    ap.add_argument("--partition", default="gpu")
    ap.add_argument("--qos", default=None)
    ap.add_argument("--projected-gb", type=float, default=200.0)
    ap.add_argument("--projected-days", type=float, default=7.0)
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).resolve().parent / "weights_manifest.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print("Unity preflight (§13.4)\n" + "-" * 72)
    check("local dataset mirrors", c_dataset_mirrors, required=False)
    check("weight checksums vs manifest", lambda: c_weight_checksums(args.manifest),
          required=False)
    check("HF_TOKEN validity", c_hf_token, required=False)
    # Required: with HF_HOME unset the encoder resolves from $HOME/.cache here
    # and the job (which sources env.sh -> workspace HF_HOME, offline) fails.
    check("caches point off $HOME", c_caches_off_home)
    check("Gate B encoder cached (offline-ready)", c_encoder_cache)
    check("committed thresholds.json resolvable", c_thresholds)
    check("ResNet-50 ImageNet weights pre-staged", c_resnet50_weights)
    check("frame stores materialised", c_frame_stores, required=False)
    check("CUDA device / VRAM / torch.version.cuda", c_cuda)
    check("REAL one-frame MuJoCo EGL render", c_egl_render)
    check("offline operation (HF_HUB_OFFLINE=1)", c_offline_ok)
    check("squota free >= 3x projected output", lambda: c_squota(args.projected_gb),
          required=False)
    check("ws_list -v remaining days / extensions",
          lambda: c_workspace(args.projected_days), required=False)
    check("$HOME quota headroom", c_home_quota, required=False)
    check("resolved --time / --qos / partition",
          lambda: c_time_and_partition(args.time, args.partition, args.qos))

    failed = [r for r in RESULTS if r["required"] and not r["passed"]]
    print("-" * 72)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed; "
          f"{len(failed)} required failures")
    if args.out:
        Path(args.out).write_text(json.dumps(RESULTS, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
