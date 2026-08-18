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
