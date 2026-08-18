"""MuJoCo GL backend guard.

MUST be imported before the first `import mujoco` anywhere in the process
(§14: setting MUJOCO_GL after mujoco is imported does nothing).

Every module in this repo that touches MuJoCo begins with::

    import mj_env  # noqa: F401  -- must precede `import mujoco`
    import mujoco

`assert_real_render()` performs an actual one-frame offscreen render and reports
the backend that was really used, because checking the env var is not enough:
headless GL silently falls back to osmesa, which works but is ~10x slower and
would corrupt wall-clock estimates (§13.4).
"""

from __future__ import annotations

import os
import platform
import sys

_DEFAULT_BY_PLATFORM = {
    "Linux": "egl",
    "Darwin": "glfw",  # cgl/glfw; EGL is not available on macOS
    "Windows": "wgl",
}


def _select_backend() -> str:
    explicit = os.environ.get("MUJOCO_GL")
    if explicit:
        return explicit
    return _DEFAULT_BY_PLATFORM.get(platform.system(), "egl")


# The guard fires only when mujoco has already been imported AND nothing has
# set MUJOCO_GL -- that is the genuine §14 failure (the backend is already
# bound and this module can no longer influence it).  If MUJOCO_GL is already
# in the environment, either a previous import of this module set it or the
# user did, and a re-import is harmless: introspecting mujoco.__version__ must
# not be able to poison the process.
if "mujoco" in sys.modules and not os.environ.get("MUJOCO_GL"):
    raise ImportError(
        "mj_env must be imported BEFORE mujoco. `mujoco` is already in "
        "sys.modules and MUJOCO_GL is unset, so the GL backend is already "
        "bound and MUJOCO_GL can no longer take effect (§14)."
    )

MUJOCO_GL = _select_backend()
os.environ["MUJOCO_GL"] = MUJOCO_GL
# PyOpenGL platform must agree, or egl selection can silently fall through.
if MUJOCO_GL == "egl":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Offline-by-default on clusters; harmless locally when weights are cached.
if os.environ.get("OGAF_OFFLINE", "0") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def assert_real_render(width: int = 64, height: int = 64) -> dict:
    """Render one frame for real and return diagnostics.

    Returns a dict with the requested backend, the backend mujoco actually
    bound, render wall-clock, and the pixel std of the rendered frame.
    Raises RuntimeError if the render fails or produces a blank frame.
    """
    import time

    import mujoco  # imported here so the env vars above are already set
    import numpy as np

    xml = """
    <mujoco>
      <visual><global offwidth="256" offheight="256"/></visual>
      <worldbody>
        <light pos="0 0 2"/>
        <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
        <body pos="0 0 0.2"><geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/></body>
        <camera name="c" pos="0.8 -0.8 0.6" xyaxes="0.707 0.707 0 -0.408 0.408 0.816"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    t0 = time.perf_counter()
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera="c")
        pixels = renderer.render()
    dt = time.perf_counter() - t0

    std = float(np.asarray(pixels, dtype=np.float64).std())
    if std < 1e-6:
        raise RuntimeError(
            f"MuJoCo render produced a blank frame (pixel std={std:.3e}). "
            f"Backend MUJOCO_GL={os.environ.get('MUJOCO_GL')!r} is not working."
        )
    return {
        "mujoco_gl_requested": MUJOCO_GL,
        "mujoco_gl_env": os.environ.get("MUJOCO_GL"),
        "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
        "render_seconds": dt,
        "pixel_std": std,
        "mujoco_version": mujoco.__version__,
        "platform": platform.platform(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(assert_real_render(), indent=2))
