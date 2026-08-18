"""Modal app: image, volume, and function definitions.  **No science inside** (§12).

Every function here is a thin wrapper that shells out to the same
substrate-agnostic scripts that run on Unity and locally.  Nothing in this file
may contain a threshold, a statistic, or an experimental decision -- if you find
yourself wanting to put one here, it belongs in the experiment script.

    modal run app.py::exp_a
    modal run app.py::corpus --geometry box --shards 0-19
    modal run app.py::exp_b
"""

from __future__ import annotations

import modal

APP_NAME = "ogaf-embodiment-shortcut"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libegl1", "libglew-dev", "libosmesa6-dev", "ffmpeg", "git")
    .pip_install(
        "mujoco", "numpy", "torch", "torchvision", "transformers", "timm",
        "imageio", "imageio-ffmpeg", "scipy", "scikit-learn", "statsmodels",
        "tqdm", "pyyaml",
    )
    # Set in the IMAGE, so it is in the environment before any `import mujoco`
    # anywhere in the process (§14).  Setting it inside a function body is too
    # late if any import has already happened.
    .env({
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "OGAF_DATA": "/data",
        "OGAF_RESULTS": "/data/results",
        "OGAF_CORPUS": "/data/corpus",
        "OGAF_CLIPS": "/data/clips",
        "OGAF_CKPT": "/data/checkpoints",
        "OGAF_THRESHOLDS": "/data/results/exp_a/thresholds.json",
        "TOKENIZERS_PARALLELISM": "false",
    })
    # Bake encoder weights at build time so runs never depend on the network.
    .run_commands(
        "python -c \"from transformers import AutoModel; "
        "AutoModel.from_pretrained('facebook/dinov2-large')\" || true"
    )
    .add_local_dir(".", remote_path="/root/ogaf", ignore=["data", "results", ".venv", ".git"])
)

vol = modal.Volume.from_name("ogaf-data", create_if_missing=True)
app = modal.App(APP_NAME, image=image)

VOLUMES = {"/data": vol}
WORKDIR = "/root/ogaf"


def _run(argv: list[str]) -> int:
    """Run a repo script in-process-ish, with the volume committed afterwards."""
    import subprocess
    import sys

    print("+", " ".join(argv), flush=True)
    r = subprocess.run([sys.executable, *argv], cwd=WORKDIR)
    vol.commit()
    if r.returncode != 0:
        raise RuntimeError(f"{argv[0]} exited {r.returncode}")
    return r.returncode


# ==========================================================================
# CPU: simulation and the state-space experiments (no GPU needed)
# ==========================================================================


@app.function(cpu=8, timeout=3600, volumes=VOLUMES)
def exp_a(n: int = 120):
    return _run(["exp_a_deltas.py", "--n", str(n)])


@app.function(cpu=8, timeout=3600, volumes=VOLUMES)
def exp_0(n: int = 700, seeds: int = 5):
    return _run(["exp_0_oracles.py", "--n", str(n), "--seeds", str(seeds)])


@app.function(cpu=8, timeout=3600, volumes=VOLUMES)
def exp_c(pairs: int = 240):
    return _run(["exp_c_injectivity.py", "--pairs", str(pairs)])


@app.function(cpu=8, timeout=3600, volumes=VOLUMES)
def corpus_shard(geometry: str, shard: int, clips: bool = False):
    """One shard, one container.

    Per-shard filenames on purpose: Modal volume writes race across parallel
    containers (§5, §14).  Merging happens in a single-container pass, never
    here.
    """
    argv = ["datasets.py", "--geometry", geometry, "--shards", str(shard)]
    if clips:
        argv.append("--clips")
    return _run(argv)


@app.function(cpu=8, timeout=3600, volumes=VOLUMES)
def validate_corpus():
    return _run(["validate_corpus.py"])


@app.function(cpu=4, timeout=3600, volumes=VOLUMES)
def exp_g(n: int = 300, k: int = 8):
    return _run(["exp_g_lipschitz.py", "--n", str(n), "--k", str(k)])


# ==========================================================================
# A10G: encoding, frozen-feature probes, masking decomposition
# ==========================================================================


@app.function(gpu="A10G", timeout=3600, volumes=VOLUMES)
def exp_b(geometry: str = "box"):
    """FIRST GPU GATE (§9-B).  Run before anything else touches a GPU."""
    return _run(["exp_b_resolution.py", "--geometry", geometry])


@app.function(gpu="A10G", timeout=7200, volumes=VOLUMES)
def train_arch_b(variant: str, seed: int, mask_mode: str = "full"):
    return _run(["train_idm.py", "--variant", variant, "--seed", str(seed),
                 "--mask-mode", mask_mode])


@app.function(gpu="A10G", timeout=7200, volumes=VOLUMES)
def masking():
    return _run(["exp_masking.py", "--train"])


# ==========================================================================
# A100: Architecture A end-to-end (also runnable on Unity -- identical script)
# ==========================================================================


@app.function(gpu="A100-40GB", timeout=7200, volumes=VOLUMES)
def train_arch_a(variant: str, seed: int, augment: bool = False):
    argv = ["train_idm.py", "--variant", variant, "--seed", str(seed)]
    if augment:
        argv.append("--augment")
    return _run(argv)


@app.function(gpu="A10G", timeout=7200, volumes=VOLUMES)
def exp_d():
    return _run(["exp_d_floors.py"])


@app.function(cpu=4, timeout=1800, volumes=VOLUMES)
def power():
    return _run(["power.py"])


@app.function(gpu="A10G", timeout=7200, volumes=VOLUMES)
def exp_e():
    """Sealed behind the §0.5 pre-registration lock; the script enforces it."""
    return _run(["exp_e_confound.py"])


@app.function(gpu="A10G", timeout=7200, volumes=VOLUMES)
def exp_f(generate: bool = False):
    argv = ["exp_f_friction.py"]
    if generate:
        argv.append("--generate")
    return _run(argv)


@app.function(cpu=4, timeout=1800, volumes=VOLUMES)
def analyze(exp: str = "unsealed"):
    return _run(["analyze.py", "--exp", exp])


# ==========================================================================
# Local entrypoints
# ==========================================================================


@app.local_entrypoint()
def stage_one():
    """Everything before the first GPU: Exp A, Exp 0 (Gate C0), Exp C (Gate C)."""
    exp_a.remote()
    exp_0.remote()
    exp_c.remote()
    analyze.remote("unsealed")


@app.local_entrypoint()
def build_corpus(geometry: str = "box", n_shards: int = 20, clips: bool = False):
    """Fan out shard generation, then merge and validate in ONE container."""
    list(corpus_shard.starmap([(geometry, s, clips) for s in range(n_shards)]))
    validate_corpus.remote()
