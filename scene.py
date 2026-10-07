"""MJCF scene, rollouts, and every §6 validator.

One scene throughout.  The arm is always present and visible: the embodiment
shortcut cannot be measured in a scene that removes the embodiment.

Layout
------
* Ground plane, configurable friction.
* One target object (box / sphere / cylinder), carrying an asymmetric,
  high-contrast cube texture so that orientation is visible in principle.
* A 4-DOF gantry pusher (x, y, z, yaw) ending in a thin paddle.  See
  ``config.ARM_DOF_DEVIATION`` for why 4 and not 3.
* A fixed camera whose pose is byte-identical in every render everywhere.

Conditions (§6.3) share the *commanded* arm trajectory ``T(a)`` exactly.  The
trajectory references the **nominal** target position, never the actual object
pose, so it is byte-identical across INTERACT / DECOY / ABSENT.

Horizons (§6.4): ``s_0`` (index 0), ``s_std`` (last step of the push),
``s_del`` (``max(settled_at, arm_rest_at)``), ``s_time`` (the same index as
``s_del`` but taken from a companion rollout whose arm withdraws by eps and
holds instead of retracting -- the T5 v3 control).
"""

from __future__ import annotations

import mj_env  # noqa: F401  -- MUST precede `import mujoco` (§14)

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

import config as C
import distances as D

__all__ = [
    "make_object_texture", "build_xml", "SceneModel", "get_scene",
    "sample_action", "sample_decoy_placement", "sample_interact_jitter",
    "tuple_streams", "approach_frame", "commanded_path", "clip_indices",
    "rollout", "max_corridor_radius", "assert_corridor_decoy_disjoint",
    "check_rest_pose_occlusion",
]

TEXTURE_PATH = C.DATA_ROOT / "assets" / "object_cube_texture.png"


# ==========================================================================
# Asset generation
# ==========================================================================


def make_object_texture(path: Path = TEXTURE_PATH, face: int = 128, force: bool = False) -> Path:
    """Write the deterministic asymmetric cube texture used by every object (§6).

    Six visually distinct faces, each with a chirality-breaking corner marker,
    laid out in MuJoCo's 3x4 ``.U..LFRB.D..`` cube grid.  Deterministic: no
    RNG, so renders are byte-reproducible across machines.
    """
    path = Path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    # Saturated, maximally separated hues.  The pattern is deliberately
    # LOW-frequency: the object subtends ~30 px in a 224x224 render, so fine
    # stripes would alias to flat grey and the texture would break visual
    # symmetry only in principle.  Four big quadrant blocks per face survive
    # that downsampling; a thick edge bar fixes chirality.
    palette = [
        (232, 32, 32),    # red
        (34, 78, 232),    # blue
        (246, 214, 24),   # yellow
        (24, 178, 96),    # green
        (246, 128, 18),   # orange
        (24, 24, 24),     # near-black
        (242, 242, 242),  # near-white
        (150, 42, 214),   # purple
    ]
    layout = ".U..LFRB.D.."   # MuJoCo 3x4 cube grid; '.' cells are unused
    order = "UFRBLD"

    img = np.zeros((3 * face, 4 * face, 3), dtype=np.uint8)
    h = face // 2
    bar = max(3, face // 10)

    for cell, letter in enumerate(layout):
        if letter == ".":
            continue
        fi = order.index(letter)
        r, c = divmod(cell, 4)
        tile = np.zeros((face, face, 3), dtype=np.uint8)
        # A different rotation of the palette per face, so no two faces match
        # and no face is invariant under a 90-degree turn.
        quad = [palette[(fi * 3 + k) % len(palette)] for k in range(4)]
        tile[:h, :h] = quad[0]
        tile[:h, h:] = quad[1]
        tile[h:, h:] = quad[2]
        tile[h:, :h] = quad[3]
        # Chirality bar: one thick edge only, on a face-dependent side.
        side = fi % 4
        if side == 0:
            tile[:bar, :] = 0
        elif side == 1:
            tile[:, -bar:] = 0
        elif side == 2:
            tile[-bar:, :] = 0
        else:
            tile[:, :bar] = 0
        img[r * face : (r + 1) * face, c * face : (c + 1) * face] = tile

    # Atomic: parallel corpus shards may all be first to need the texture, and
    # MuJoCo must never read a half-written PNG.  Same suffix so the format is
    # inferred; os.replace is atomic on POSIX.
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    imageio.imwrite(tmp, img)
    os.replace(tmp, path)
    return path


# ==========================================================================
# MJCF
# ==========================================================================


def _camera_xml() -> str:
    return (
        f'<camera name="{C.CAMERA_NAME}" mode="targetbody" target="cam_target" '
        f'pos="{C.CAMERA_POS[0]} {C.CAMERA_POS[1]} {C.CAMERA_POS[2]}" '
        f'fovy="{C.CAMERA_FOVY}"/>'
    )


def _object_geom_xml(geometry: str, friction_mult: float) -> str:
    fs, ft, fr = C.OBJECT_FRICTION_BY_GEOMETRY[geometry]
    friction = f"{fs * friction_mult} {ft} {fr}"
    size = C.GEOM_SIZE[geometry]
    size_str = " ".join(str(s) for s in size)
    return (
        f'<geom name="obj" type="{geometry}" size="{size_str}" material="matobj" '
        f'density="{C.OBJECT_DENSITY}" friction="{friction}" priority="1" '
        f'contype="4" conaffinity="3" condim="6" solref="0.005 1"/>'
    )


def object_rest_height(geometry: str) -> float:
    """z of the object's body frame when it rests on the plane."""
    if geometry == "box":
        return C.GEOM_SIZE["box"][2]
    if geometry == "sphere":
        return C.GEOM_SIZE["sphere"][0]
    if geometry == "cylinder":
        return C.GEOM_SIZE["cylinder"][1]
    raise ValueError(geometry)


def build_xml(geometry: str, condition: str, friction_mult: float = 1.0,
              texture_path: Path | None = None) -> str:
    """MJCF for one (geometry, condition).

    ABSENT compiles a model with no object body at all, rather than hiding one:
    a far-away body still renders as sub-pixel speckle and still occupies qpos,
    both of which are silent failure modes.
    """
    if condition not in C.CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if geometry not in C.GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}")
    tex = Path(texture_path) if texture_path is not None else make_object_texture()

    px, py, pz = C.PADDLE_HALF_EXTENTS
    pf_s, pf_t, pf_r = C.PLANE_FRICTION

    object_body = ""
    if condition != "ABSENT":
        object_body = f"""
    <body name="object" pos="0 0 {object_rest_height(geometry)}">
      <freejoint name="obj_free"/>
      {_object_geom_xml(geometry, friction_mult)}
    </body>"""

    if C.ARM_CONTROL == "position":
        actuators = "\n".join(
            f'    <position name="act_{j}" joint="{j}" kp="{C.ARM_KP}" '
            f'dampratio="1"/>' for j in C.ARM_JOINTS
        )
        actuator_block = f"  <actuator>\n{actuators}\n  </actuator>"
        damping = C.ARM_DAMPING
    else:
        actuator_block = ""
        damping = 0.0

    return f"""
<mujoco model="ogaf_push_{geometry}_{condition}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{C.SIM_DT}" integrator="implicitfast" cone="elliptic"
          impratio="10" iterations="100" tolerance="1e-10"/>
  <visual>
    <global offwidth="{C.RENDER_SIZE}" offheight="{C.RENDER_SIZE}"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
  </visual>

  <asset>
    <texture name="skytex" type="skybox" builtin="gradient"
             rgb1="0.25 0.32 0.42" rgb2="0.06 0.08 0.11" width="256" height="256"/>
    <texture name="gridtex" type="2d" builtin="checker"
             rgb1="0.34 0.34 0.37" rgb2="0.46 0.46 0.49" width="512" height="512"/>
    <material name="matplane" texture="gridtex" texuniform="true" texrepeat="4 4"
              reflectance="0.02"/>
    <texture name="objtex" type="cube" file="{tex}" gridsize="3 4"
             gridlayout=".U..LFRB.D.."/>
    <material name="matobj" texture="objtex" specular="0.15" shininess="0.1"/>
    <material name="matarm" rgba="0.88 0.88 0.92 1" specular="0.4" shininess="0.4"/>
  </asset>

  <worldbody>
    <light name="key" pos="0.6 -0.4 1.4" dir="-0.35 0.25 -1" directional="true"
           diffuse="0.6 0.6 0.6" specular="0.15 0.15 0.15"/>
    <light name="fill" pos="-0.6 0.6 1.0" dir="0.4 -0.4 -1" directional="true"
           diffuse="0.25 0.25 0.25" specular="0 0 0"/>

    <geom name="floor" type="plane" size="4 4 0.05" material="matplane"
          friction="{pf_s} {pf_t} {pf_r}" contype="1" conaffinity="4"
          condim="6" solref="0.005 1"/>

    <body name="cam_target" pos="{C.CAMERA_TARGET[0]} {C.CAMERA_TARGET[1]} {C.CAMERA_TARGET[2]}">
      <site name="camsite" size="0.0005" rgba="0 0 0 0"/>
    </body>
    {_camera_xml()}

    <body name="arm" pos="0 0 0">
      <joint name="arm_x" type="slide" axis="1 0 0" range="-0.9 0.9" damping="{damping}"/>
      <joint name="arm_y" type="slide" axis="0 1 0" range="-0.9 0.9" damping="{damping}"/>
      <joint name="arm_z" type="slide" axis="0 0 1" range="-0.05 0.70" damping="{damping}"/>
      <joint name="arm_yaw" type="hinge" axis="0 0 1" range="-3.2 3.2" damping="{damping}"/>
      <geom name="paddle" type="box" size="{px} {py} {pz}" pos="0 0 0" material="matarm"
            contype="2" conaffinity="4" condim="6" solref="0.005 1"/>
      <geom name="stalk" type="capsule" fromto="0 0 {pz} 0 0 {C.STALK_TOP_REL}"
            size="{C.STALK_RADIUS}" material="matarm" contype="0" conaffinity="0"/>
      <geom name="wrist" type="box" size="{px * 2} {py * 0.5} {py * 0.5}"
            pos="{-px * 3} 0 {pz * 0.6}" material="matarm" contype="0" conaffinity="0"/>
    </body>
{object_body}
  </worldbody>

{actuator_block}
</mujoco>
""".strip()


# ==========================================================================
# Compiled-model cache
# ==========================================================================


@dataclass
class SceneModel:
    model: mujoco.MjModel
    geometry: str
    condition: str
    friction_mult: float
    arm_qadr: np.ndarray      # qpos addresses of the 4 arm joints, in ARM_JOINTS order
    arm_vadr: np.ndarray      # qvel addresses
    obj_qadr: int | None      # qpos address of the free joint, or None for ABSENT
    obj_vadr: int | None
    obj_bid: int | None
    obj_gid: int | None
    paddle_gid: int
    arm_gids: tuple
    xml_sha256: str


_SCENE_CACHE: dict = {}


def get_scene(geometry: str, condition: str, friction_mult: float = 1.0) -> SceneModel:
    key = (geometry, condition, round(float(friction_mult), 9), C.ARM_CONTROL)
    if key in _SCENE_CACHE:
        return _SCENE_CACHE[key]

    xml = build_xml(geometry, condition, friction_mult)
    model = mujoco.MjModel.from_xml_string(xml)

    def jid(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if i < 0:
            raise RuntimeError(f"joint {name!r} not found")
        return i

    def gid(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if i < 0:
            raise RuntimeError(f"geom {name!r} not found")
        return i

    arm_j = [jid(n) for n in C.ARM_JOINTS]
    arm_qadr = np.array([model.jnt_qposadr[i] for i in arm_j], dtype=np.int64)
    arm_vadr = np.array([model.jnt_dofadr[i] for i in arm_j], dtype=np.int64)

    if condition == "ABSENT":
        obj_qadr = obj_vadr = obj_bid = obj_gid = None
    else:
        oj = jid("obj_free")
        obj_qadr = int(model.jnt_qposadr[oj])
        obj_vadr = int(model.jnt_dofadr[oj])
        obj_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object"))
        obj_gid = gid("obj")

    sm = SceneModel(
        model=model,
        geometry=geometry,
        condition=condition,
        friction_mult=float(friction_mult),
        arm_qadr=arm_qadr,
        arm_vadr=arm_vadr,
        obj_qadr=obj_qadr,
        obj_vadr=obj_vadr,
        obj_bid=obj_bid,
        obj_gid=obj_gid,
        paddle_gid=gid("paddle"),
        arm_gids=(gid("paddle"), gid("stalk"), gid("wrist")),
        xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
    )
    _SCENE_CACHE[key] = sm
    return sm


# ==========================================================================
# Sampling -- independent RNG streams (T11)
# ==========================================================================


def tuple_streams(master_seed: int, geometry: str, tuple_index: int) -> dict:
    """Provably independent RNG streams for one (action, seed) tuple.

    The action stream and the placement/jitter streams are *separate spawns* of
    a common SeedSequence, so DECOY placement cannot correlate with the action
    even by accident.  This is the structural half of the T11 countermeasure;
    ``validate_corpus.py`` runs the empirical ridge-regression half.
    """
    geom_ord = C.GEOMETRIES.index(geometry)
    ss = np.random.SeedSequence([master_seed, geom_ord, tuple_index])
    action_ss, decoy_ss, jitter_ss = ss.spawn(3)
    return {
        "action": np.random.default_rng(action_ss),
        "decoy": np.random.default_rng(decoy_ss),
        "jitter": np.random.default_rng(jitter_ss),
        "entropy": [master_seed, geom_ord, tuple_index],
    }


def sample_action(rng: np.random.Generator) -> np.ndarray:
    lo, hi = C.action_ranges_array()
    return rng.uniform(lo, hi)


def sample_decoy_placement(rng: np.random.Generator) -> np.ndarray:
    """(x, y, yaw) for the DECOY object -- independent of the action by construction."""
    r_in, r_out = C.DECOY_ANNULUS
    # area-uniform in the annulus
    r = math.sqrt(rng.uniform(r_in ** 2, r_out ** 2))
    phi = rng.uniform(-math.pi, math.pi)
    yaw = rng.uniform(-math.pi, math.pi)
    return np.array([r * math.cos(phi), r * math.sin(phi), yaw])


def sample_interact_jitter(rng: np.random.Generator) -> np.ndarray:
    """(dx, dy, yaw) jitter of the INTERACT object's initial pose."""
    j = C.INTERACT_POS_JITTER
    dx, dy = rng.uniform(-j, j, size=2)
    yaw = rng.uniform(-math.pi, math.pi) if C.INTERACT_YAW_JITTER else 0.0
    return np.array([dx, dy, yaw])


# ==========================================================================
# Commanded trajectory  T(a)
# ==========================================================================


def approach_frame(theta: float) -> tuple[np.ndarray, np.ndarray]:
    """(u, n): unit approach direction and its left-hand normal, in the ground plane."""
    bx, by = C.BASE_APPROACH_DIR
    c, s = math.cos(theta), math.sin(theta)
    u = np.array([c * bx - s * by, s * bx + c * by])
    u = u / np.linalg.norm(u)
    n = np.array([-u[1], u[0]])
    return u, n


def _minjerk(t: np.ndarray) -> np.ndarray:
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def _interp(p0, p1, n_steps: int) -> np.ndarray:
    """Min-jerk interpolation, n_steps rows, ending exactly at p1."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    tau = (np.arange(1, n_steps + 1) / n_steps)
    s = _minjerk(tau)[:, None]
    return p0[None, :] + s * (p1 - p0)[None, :]


def commanded_path(action, arm_variant: str = "retract",
                   phases: C.PhaseSteps | None = None) -> np.ndarray:
    """(T, 4) commanded arm configuration (x, y, z, yaw) for every step.

    Depends only on the action, never on the object -- so it is byte-identical
    across INTERACT / DECOY / ABSENT (T9).

    ``arm_variant``:
      * ``"retract"``   -- the s_del arm: retreat, lift, return to canonical rest.
      * ``"eps_hold"``  -- the s_time arm (T5 v3 fix): withdraw by
        ``EPS_WITHDRAW`` along -u right after the push, then hold.  Holding the
        arm *in* its final contact pose would change the physics (the settling
        object can rebound off it); an eps-withdrawal removes contact while
        leaving elapsed time and arm visibility matched.
    """
    if arm_variant not in ("retract", "eps_hold"):
        raise ValueError(arm_variant)
    ph = phases or C.PHASES
    v, theta, d = (float(x) for x in action)
    u, n = approach_frame(theta)

    rest = np.array([C.REST_POSE[0], C.REST_POSE[1], C.REST_POSE[2], C.REST_YAW])
    start_xy = -C.PUSH_START_RADIUS * u + d * n           # nominal target is the origin
    above = np.array([start_xy[0], start_xy[1], C.REST_POSE[2], theta])
    at_push = np.array([start_xy[0], start_xy[1], C.PUSH_HEIGHT, theta])

    segs = [np.repeat(rest[None, :], ph.hold0, axis=0)]
    segs.append(_interp(rest, above, ph.traverse))
    segs.append(_interp(above, at_push, ph.descend))

    # Constant-velocity push: the commanded speed *is* the action, so no easing.
    k = np.arange(1, ph.push + 1)
    push_xy = start_xy[None, :] + (v * k * C.SIM_DT)[:, None] * u[None, :]
    push = np.concatenate(
        [push_xy, np.full((ph.push, 1), C.PUSH_HEIGHT), np.full((ph.push, 1), theta)], axis=1
    )
    segs.append(push)
    end = push[-1].copy()

    # Both variants share the eps-withdrawal.  This matters: contact typically
    # persists to the end of the push, so if the two arms parted company at
    # push_end they would apply *different* contact forces during the final
    # milliseconds of contact and settle the object in measurably different
    # places -- which is precisely what the T5 settled-pose-match validator
    # then rejects.  Sharing the withdrawal makes the arms identical until
    # contact is provably broken, and only then do they diverge.
    wd_xy = end[:2] - C.EPS_WITHDRAW * u
    wd = np.array([wd_xy[0], wd_xy[1], C.PUSH_HEIGHT, theta])
    segs.append(_interp(end, wd, ph.eps_withdraw))

    if arm_variant == "retract":
        retreat_xy = wd[:2] - (C.RETREAT_DISTANCE - C.EPS_WITHDRAW) * u
        retreat = np.array([retreat_xy[0], retreat_xy[1], C.PUSH_HEIGHT, theta])
        segs.append(_interp(wd, retreat, ph.retreat - ph.eps_withdraw))
        # Lift straight up BEFORE travelling home, mirroring the approach.
        # Interpolating xy and z together sweeps the paddle diagonally across
        # the table, and when the object has settled between the retreat point
        # and the rest pose the paddle clips it -- re-contacting the object long
        # after the push, at a time that differs between the s_del and s_time
        # arms.  That showed up as ~10% of box rollouts failing the T5
        # settled-pose match by tens of millimetres.
        n_lift = max(1, ph.lift_home // 3)
        n_home = ph.lift_home - n_lift
        lifted = np.array([retreat[0], retreat[1], C.REST_POSE[2], theta])
        segs.append(_interp(retreat, lifted, n_lift))
        segs.append(_interp(lifted, rest, n_home))
        tail = ph.max_steps - sum(s.shape[0] for s in segs)
        segs.append(np.repeat(rest[None, :], tail, axis=0))
    else:
        tail = ph.max_steps - sum(s.shape[0] for s in segs)
        segs.append(np.repeat(wd[None, :], tail, axis=0))

    path = np.concatenate(segs, axis=0)
    assert path.shape == (ph.max_steps, 4), path.shape
    return path


def max_corridor_radius(n_grid: int = 4001) -> float:
    """Largest distance from the nominal target reached by any commanded paddle point.

    Used to prove the DECOY annulus is disjoint from every possible push
    corridor, which is what lets DECOY placement be sampled with *no*
    action-dependent rejection (the structural T11 defence).
    """
    lo, hi = C.action_ranges_array()
    rng = np.random.default_rng(0)
    # The extremum is on the boundary of the action box; sample the box densely
    # and include all 8 corners explicitly.
    corners = np.array(np.meshgrid(*[[lo[i], hi[i]] for i in range(3)])).reshape(3, -1).T
    samples = np.concatenate([corners, rng.uniform(lo, hi, size=(n_grid, 3))], axis=0)
    px, py, _ = C.PADDLE_HALF_EXTENTS
    half_diag = math.hypot(px, py)
    best = 0.0
    ph = C.PHASES
    lo_i, hi_i = ph.push_start_idx, ph.push_end_idx + ph.retreat + 1
    for a in samples:
        path = commanded_path(a, "retract")
        # Only the low-altitude portion can touch a resting object.
        seg = path[lo_i:hi_i, :2]
        best = max(best, float(np.max(np.linalg.norm(seg, axis=1))) + half_diag)
    return best


def assert_corridor_decoy_disjoint(geometry: str, margin: float = 0.005) -> dict:
    """ASSERT: no reachable push corridor can reach the DECOY annulus (T11)."""
    corridor = max_corridor_radius()
    obj_reach = max(C.GEOM_SIZE[geometry]) * math.sqrt(3)  # worst-case circumscribed radius
    nearest_decoy_surface = C.DECOY_ANNULUS[0] - obj_reach
    clearance = nearest_decoy_surface - corridor
    passed = clearance > margin
    return {
        "name": "T11_corridor_decoy_disjoint",
        "passed": bool(passed),
        "corridor_radius_m": corridor,
        "nearest_decoy_surface_m": nearest_decoy_surface,
        "clearance_m": clearance,
        "threshold_m": margin,
        "geometry": geometry,
    }


def clip_indices(variant: str, s_del_idx: int, k: int | None = None,
                 phases: C.PhaseSteps | None = None) -> np.ndarray:
    """Frame indices for the clip-input variants (§8.1).

    ``A-std``      : k frames spanning [0, push_end]  -- the action-execution window.
    ``A-clip-del`` : k-1 frames spanning [0, arm_rest] plus the actual s_del frame.

    Both index sets are deterministic given the variant (``arm_rest`` and
    ``push_end`` are fixed by the phase schedule), so clips can be rendered in a
    single simulation pass.
    """
    ph = phases or C.PHASES
    k = k or C.CLIP_LENGTH
    if variant == "A-std":
        return np.unique(np.linspace(0, ph.push_end_idx, k).round().astype(np.int64))
    if variant == "A-clip-del":
        # Fixed length by construction.  The head ends one step BEFORE arm
        # rest and s_del >= arm_rest_idx always (see rollout), so the s_del
        # frame can never collide with the head and every clip has exactly k
        # frames -- a variable-length clip would break batch collation.
        head = np.linspace(0, ph.arm_rest_idx - 1, k - 1).round().astype(np.int64)
        idx = np.concatenate([head, [int(s_del_idx)]]).astype(np.int64)
        if len(np.unique(idx)) != k or not (np.diff(idx) > 0).all():
            raise ValueError(f"A-clip-del clip is not {k} strictly increasing frames "
                             f"(s_del_idx={s_del_idx}, arm_rest_idx={ph.arm_rest_idx})")
        return idx
    raise ValueError(f"{variant!r} is not a clip variant; see config.CLIP_VARIANTS")


# ==========================================================================
# Simulation
# ==========================================================================


def _contact_force_between(model, data, gid_a: int, gid_b: int) -> float:
    """Total contact-force magnitude between two geoms at the current state."""
    total = 0.0
    buf = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        con = data.contact[i]
        if (con.geom1 == gid_a and con.geom2 == gid_b) or (
            con.geom1 == gid_b and con.geom2 == gid_a
        ):
            mujoco.mj_contactForce(model, data, i, buf)
            total += float(np.linalg.norm(buf[:3]))
    return total


def _simulate(sm: SceneModel, path: np.ndarray, obj_init: np.ndarray | None,
              *, disable_arm_object_contact: bool = False):
    """Run one arm variant.  Returns trajectories; no rendering.

    State at index ``t`` is *recorded before* the step that advances to ``t+1``,
    and under ``ARM_CONTROL == "kinematic"`` the arm's qpos/qvel are re-clamped
    to the command immediately after every step.  Consequently
    ``arm_qpos[t] == path[t]`` **exactly**, for every condition -- which is the
    T9 guarantee (quasi-infinite arm impedance) made structural rather than
    merely measured.  The object still receives the correct contact impulse,
    because the clamp happens after ``mj_step`` has already integrated it.

    Under ``ARM_CONTROL == "position"`` nothing is clamped and the recorded
    trajectory is the true PD response, which the T9 validator then measures
    against a contact-free reference.

    ``disable_arm_object_contact`` builds the contact-free reference used by
    that measurement.
    """
    model, data = sm.model, mujoco.MjData(sm.model)
    mujoco.mj_resetData(model, data)
    T = path.shape[0]

    saved_contype = None
    if disable_arm_object_contact:
        saved_contype = int(model.geom_contype[sm.paddle_gid])
        model.geom_contype[sm.paddle_gid] = 0

    try:
        data.qpos[sm.arm_qadr] = path[0]
        data.qvel[sm.arm_vadr] = 0.0
        if sm.obj_qadr is not None:
            assert obj_init is not None, "object condition needs an initial pose"
            data.qpos[sm.obj_qadr : sm.obj_qadr + 7] = obj_init
            data.qvel[sm.obj_vadr : sm.obj_vadr + 6] = 0.0
        if C.ARM_CONTROL == "position":
            data.ctrl[:] = path[0]
        mujoco.mj_forward(model, data)

        arm_qpos = np.empty((T, 4), dtype=np.float64)
        obj_poses = np.empty((T, 7), dtype=np.float64) if sm.obj_qadr is not None else None
        obj_vel = np.empty((T, 6), dtype=np.float64) if sm.obj_qadr is not None else None
        contact_forces = np.zeros(T, dtype=np.float64)
        states = np.empty((T, model.nq), dtype=np.float64)

        for t in range(T):
            # ---- record the state AT index t ----
            arm_qpos[t] = data.qpos[sm.arm_qadr]
            states[t] = data.qpos
            if sm.obj_qadr is not None:
                obj_poses[t] = data.qpos[sm.obj_qadr : sm.obj_qadr + 7]
                obj_vel[t] = data.qvel[sm.obj_vadr : sm.obj_vadr + 6]
                contact_forces[t] = _contact_force_between(
                    model, data, sm.paddle_gid, sm.obj_gid
                )
            if t + 1 == T:
                break

            # ---- advance to t+1 ----
            if C.ARM_CONTROL == "kinematic":
                data.qvel[sm.arm_vadr] = (path[t + 1] - path[t]) / C.SIM_DT
            else:
                data.ctrl[:] = path[t + 1]
            mujoco.mj_step(model, data)
            if C.ARM_CONTROL == "kinematic":
                data.qpos[sm.arm_qadr] = path[t + 1]
                data.qvel[sm.arm_vadr] = (path[t + 1] - path[t]) / C.SIM_DT
                mujoco.mj_forward(model, data)
    finally:
        if saved_contype is not None:
            model.geom_contype[sm.paddle_gid] = saved_contype

    return {
        "arm_qpos": arm_qpos,
        "arm_cmd": path.astype(np.float64, copy=True),
        "obj_poses": obj_poses,
        "obj_vel": obj_vel,
        "contact_forces": contact_forces,
        "qpos_traj": states,
    }


def _render_at(sm: SceneModel, qpos_traj: np.ndarray, indices, renderer, seg_renderer):
    """Render RGB (and segmentation) at the given step indices."""
    data = mujoco.MjData(sm.model)
    frames, masks = [], []
    for idx in indices:
        data.qpos[:] = qpos_traj[int(idx)]
        data.qvel[:] = 0.0
        mujoco.mj_forward(sm.model, data)
        renderer.update_scene(data, camera=C.CAMERA_NAME)
        frames.append(renderer.render().copy())
        if seg_renderer is not None:
            seg_renderer.update_scene(data, camera=C.CAMERA_NAME)
            raw = seg_renderer.render()
            masks.append(_seg_to_labels(sm, raw))
    return frames, masks


def _seg_to_labels(sm: SceneModel, raw: np.ndarray) -> np.ndarray:
    """MuJoCo segmentation output -> {0 background, 1 arm, 2 object}."""
    gids = raw[..., 0]
    types = raw[..., 1]
    out = np.zeros(gids.shape, dtype=np.uint8)
    is_geom = types == int(mujoco.mjtObj.mjOBJ_GEOM)
    for g in sm.arm_gids:
        out[is_geom & (gids == g)] = C.SEG_ARM
    if sm.obj_gid is not None:
        out[is_geom & (gids == sm.obj_gid)] = C.SEG_OBJECT
    return out


# ==========================================================================
# Horizon detection
# ==========================================================================


def compute_contact_end(contact_forces: np.ndarray, zero_tol: float) -> int:
    """Last step with nonzero arm-object contact force; -1 if never in contact."""
    nz = np.nonzero(contact_forces > zero_tol)[0]
    return int(nz[-1]) if nz.size else -1


def compute_settled_at(obj_vel: np.ndarray, start_idx: int,
                       lin_thresh: float, ang_thresh: float,
                       hold_steps: int | None = None) -> int | None:
    """First step after ``start_idx`` where the object is below both speed
    thresholds and *stays* below for ``hold_steps`` (a single quiet step can be
    a turning point of a bounce)."""
    if obj_vel is None:
        return start_idx
    if hold_steps is None:
        hold_steps = 2 * C.FRAME_STRIDE  # quiet for two video frames, not one
    lin = np.linalg.norm(obj_vel[:, :3], axis=1)
    ang = np.linalg.norm(obj_vel[:, 3:], axis=1)
    quiet = (lin < lin_thresh) & (ang < ang_thresh)
    n = quiet.size
    for t in range(max(start_idx, 0), n - hold_steps):
        if quiet[t : t + hold_steps].all():
            return int(t)
    return None


# ==========================================================================
# In-frustum check
# ==========================================================================


def _project(model, cam_id: int, data, points: np.ndarray) -> np.ndarray:
    """World points -> pixel coordinates for the fixed camera."""
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    rel = (points - cam_pos) @ cam_mat            # camera frame: -z forward
    fovy = math.radians(model.cam_fovy[cam_id])
    f = 0.5 * C.RENDER_SIZE / math.tan(fovy / 2.0)
    z = -rel[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        px = f * rel[:, 0] / z + C.RENDER_SIZE / 2.0
        py = -f * rel[:, 1] / z + C.RENDER_SIZE / 2.0
    px = np.where(z > 1e-6, px, np.nan)
    py = np.where(z > 1e-6, py, np.nan)
    return np.stack([px, py], axis=1)


def _object_corner_points(geometry: str, pose: np.ndarray) -> np.ndarray:
    """Bounding points of the object at ``pose`` (7-vector), in world coordinates."""
    R = D.quat_to_matrix(pose[3:])
    p = pose[:3]
    s = C.GEOM_SIZE[geometry]
    if geometry == "sphere":
        r = s[0]
        offs = np.array([[dx, dy, dz] for dx in (-r, r) for dy in (-r, r) for dz in (-r, r)])
    elif geometry == "box":
        offs = np.array([[dx, dy, dz] for dx in (-s[0], s[0])
                         for dy in (-s[1], s[1]) for dz in (-s[2], s[2])])
    else:  # cylinder
        r, h = s
        offs = np.array([[dx, dy, dz] for dx in (-r, r) for dy in (-r, r) for dz in (-h, h)])
    return p[None, :] + offs @ R.T


# ==========================================================================
# rollout()
# ==========================================================================


def rollout(action, seed: int, condition: str, geometry: str,
            retract: bool = True, friction_mult: float = 1.0,
            *, thresholds: dict | None = None,
            render: bool = True, clip_variant: str | None = None,
            tolerances: C.Tolerances | None = None,
            master_seed: int | None = None) -> dict:
    """Simulate one (action, seed, condition, geometry) rollout.

    Returns the dict documented in §6.4.  Every T5 / T9 / T10 / T11 assertion
    lives in ``validators`` with its *measured value*, so a finished corpus can
    be re-checked offline by ``validate_corpus.py`` without re-simulating (§0.4).

    ``thresholds`` is the per-geometry dict from ``config.load_thresholds()``.
    It may be None only for Experiment A, which is what *produces* those
    thresholds; then ``settled_at`` is left None and s_del falls back to
    ``arm_rest_at``.
    """
    action = np.asarray(action, dtype=np.float64).reshape(3)
    tol = tolerances or C.TOLERANCES
    ms = C.MASTER_SEED if master_seed is None else master_seed
    ph = C.PHASES
    sm = get_scene(geometry, condition, friction_mult)

    streams = tuple_streams(ms, geometry, int(seed))
    obj_init = None
    decoy_xy = None
    if condition == "INTERACT":
        dx, dy, yaw = sample_interact_jitter(streams["jitter"])
        q = D.quat_from_axis_angle([0, 0, 1], yaw)
        obj_init = np.concatenate([[dx, dy, object_rest_height(geometry)], q])
    elif condition == "DECOY":
        x, y, yaw = sample_decoy_placement(streams["decoy"])
        decoy_xy = np.array([x, y])
        q = D.quat_from_axis_angle([0, 0, 1], yaw)
        obj_init = np.concatenate([[x, y, object_rest_height(geometry)], q])

    primary_variant = "retract" if retract else "eps_hold"
    path_main = commanded_path(action, primary_variant, ph)
    sim_main = _simulate(sm, path_main, obj_init)

    zero_tol = tol.contact_force_zero_n
    contact_end = compute_contact_end(sim_main["contact_forces"], zero_tol)

    settled_at = None
    if thresholds is not None and sim_main["obj_vel"] is not None:
        settled_at = compute_settled_at(
            sim_main["obj_vel"],
            start_idx=max(contact_end, ph.push_end_idx) + 1,
            lin_thresh=thresholds["settle_lin_speed"],
            ang_thresh=thresholds["settle_ang_speed"],
        )
    arm_rest_at = ph.arm_rest_idx
    if retract:
        s_del_idx = max(settled_at, arm_rest_at) if settled_at is not None else arm_rest_at
    else:
        s_del_idx = settled_at if settled_at is not None else arm_rest_at
    s_del_idx = int(min(s_del_idx, ph.max_steps - 1))

    horizon_idx = {
        "s_0": 0,
        "s_std": ph.push_end_idx,
        "s_del": s_del_idx,
        "s_time": s_del_idx,
    }

    # --- companion eps-hold rollout for s_time (T5) -----------------------
    sim_time = None
    if retract:
        path_time = commanded_path(action, "eps_hold", ph)
        sim_time = _simulate(sm, path_time, obj_init)

    # --- contact-free arm reference for the T9 measurement ---------------
    # T9 asks for max||arm_qpos_INTERACT - arm_qpos_DECOY|| over the whole
    # rollout.  DECOY is by definition contact-free, so a same-action rollout
    # with arm-object contact disabled is an exact stand-in and does not
    # require the paired rollout to be materialised here.
    arm_ref = None
    if condition == "INTERACT" and C.ARM_CONTROL == "position":
        arm_ref = _simulate(sm, path_main, obj_init,
                            disable_arm_object_contact=True)["arm_qpos"]

    # --- rendering --------------------------------------------------------
    frames, seg_masks, clip = {}, {}, None
    clip_idx = None
    clips: dict = {}
    # One clip per requested variant, all rendered from the same simulation
    # pass.  A-std and A-clip-del sample DIFFERENT frames (config.CLIP_VARIANTS),
    # so the corpus must store both or the T8 ablation is vacuous.
    clip_variants = ([] if clip_variant is None
                     else [clip_variant] if isinstance(clip_variant, str)
                     else list(clip_variant))
    if render:
        with mujoco.Renderer(sm.model, height=C.RENDER_SIZE, width=C.RENDER_SIZE) as rgb_r, \
             mujoco.Renderer(sm.model, height=C.RENDER_SIZE, width=C.RENDER_SIZE) as seg_r:
            seg_r.enable_segmentation_rendering()
            main_keys = ["s_0", "s_std", "s_del"] if retract else ["s_0", "s_std", "s_time"]
            f, m = _render_at(sm, sim_main["qpos_traj"],
                              [horizon_idx[k] for k in main_keys], rgb_r, seg_r)
            for k, fr, mk in zip(main_keys, f, m):
                frames[k], seg_masks[k] = fr, mk
            if sim_time is not None:
                f2, m2 = _render_at(sm, sim_time["qpos_traj"], [s_del_idx], rgb_r, seg_r)
                frames["s_time"], seg_masks["s_time"] = f2[0], m2[0]
            for cv in clip_variants:
                ci = clip_indices(cv, s_del_idx, phases=ph)
                cf, _ = _render_at(sm, sim_main["qpos_traj"], ci, rgb_r, None)
                clips[cv] = (np.stack(cf), ci)
            if clip_variants:
                clip, clip_idx = clips[clip_variants[0]]

    # --- validators -------------------------------------------------------
    validators = _run_validators(
        sm=sm, geometry=geometry, condition=condition, action=action,
        sim_main=sim_main, sim_time=sim_time, horizon_idx=horizon_idx,
        contact_end=contact_end, settled_at=settled_at, arm_rest_at=arm_rest_at,
        seg_masks=seg_masks, tol=tol, thresholds=thresholds, retract=retract,
        path_main=path_main, arm_ref=arm_ref,
    )

    return {
        "action": action,
        "seed": int(seed),
        "condition": condition,
        "geometry": geometry,
        "friction_mult": float(friction_mult),
        "retract": bool(retract),
        "obj_poses": sim_main["obj_poses"],
        "obj_vel": sim_main["obj_vel"],
        "arm_qpos": sim_main["arm_qpos"],
        "arm_cmd": sim_main["arm_cmd"],
        "obj_poses_time": None if sim_time is None else sim_time["obj_poses"],
        "arm_qpos_time": None if sim_time is None else sim_time["arm_qpos"],
        "contact_forces": sim_main["contact_forces"],
        "contact_forces_time": None if sim_time is None else sim_time["contact_forces"],
        "frames": frames,
        "seg_masks": seg_masks,
        "clip": clip,
        "clip_indices": clip_idx,
        "clip_variant": clip_variants[0] if clip_variants else None,
        "clips": clips,
        "decoy_xy": decoy_xy,
        "obj_init": obj_init,
        "contact_end": int(contact_end),
        "settled_at": None if settled_at is None else int(settled_at),
        "arm_rest_at": int(arm_rest_at),
        "horizon_idx": horizon_idx,
        "xml_sha256": sm.xml_sha256,
        "validators": validators,
    }


# ==========================================================================
# Validators (§6, T5 / T9 / T10)
# ==========================================================================


def _v(name, passed, value, threshold, **extra):
    d = {"name": name, "passed": bool(passed), "value": None if value is None else float(value),
         "threshold": None if threshold is None else float(threshold)}
    d.update(extra)
    return d


def _run_validators(*, sm, geometry, condition, action, sim_main, sim_time,
                    horizon_idx, contact_end, settled_at, arm_rest_at,
                    seg_masks, tol, thresholds, retract, path_main,
                    arm_ref=None) -> dict:
    out: dict = {}
    ph = C.PHASES

    # -- T10: contact presence / absence ---------------------------------
    if condition == "INTERACT":
        forces = sim_main["contact_forces"]
        hit = forces > tol.contact_force_zero_n
        contact_start = int(np.argmax(hit)) if hit.any() else -1
        out["T10_interact_contact"] = _v(
            "T10_interact_contact", contact_end > 0, contact_end, 0.0,
            note="INTERACT rollouts must actually make contact; misses are excluded and counted",
            peak_force_n=float(np.max(forces)), contact_start=contact_start,
        )
        # s_std must genuinely be a post-contact frame, or the "standard
        # horizon" is not the horizon the critiqued protocols use.
        out["T10_contact_before_sstd"] = _v(
            "T10_contact_before_sstd",
            0 <= contact_start <= ph.push_end_idx,
            contact_start, ph.push_end_idx,
            note="first contact must occur at or before s_std",
        )
    elif condition == "DECOY":
        peak = float(np.max(sim_main["contact_forces"])) if sim_main["contact_forces"].size else 0.0
        out["T10_decoy_no_contact"] = _v(
            "T10_decoy_no_contact", peak <= tol.contact_force_zero_n,
            peak, tol.contact_force_zero_n,
            note="accidental grazes from placement-range clipping must be zero",
        )

    # -- T9: the executed arm trajectory must not depend on the condition ---
    # The measured quantity is max||arm_qpos(with contact) - arm_qpos(without
    # contact)|| over the ENTIRE rollout (not just pre-contact), which is
    # exactly the INTERACT-vs-DECOY arm residual the spec asks for.
    if arm_ref is not None:
        dev = float(np.max(np.linalg.norm(sim_main["arm_qpos"][:, :3] - arm_ref[:, :3], axis=1)))
        dev_yaw = float(np.max(np.abs(sim_main["arm_qpos"][:, 3] - arm_ref[:, 3])))
        basis = "contact_free_reference_simulation"
    else:
        # Kinematic (quasi-infinite impedance) arm, or a contact-free
        # condition: the executed trajectory is the commanded one, so the
        # cross-condition residual is the command-tracking residual, which is
        # identically zero by construction.  Verified, not assumed.
        dev = float(np.max(np.linalg.norm(sim_main["arm_qpos"][:, :3] - path_main[:, :3], axis=1)))
        dev_yaw = float(np.max(np.abs(sim_main["arm_qpos"][:, 3] - path_main[:, 3])))
        basis = "commanded_path (arm is kinematically clamped)"
    within = dev <= tol.arm_deviation_m
    out["T9_arm_deviation"] = _v(
        "T9_arm_deviation", within or not C.T9_EXCLUDES, dev, tol.arm_deviation_m,
        exceeds_tolerance=not within, excludes=C.T9_EXCLUDES,
        yaw_deviation_rad=dev_yaw, arm_control=C.ARM_CONTROL, basis=basis,
        note="residual action information carried by the arm reaction channel; "
             "Exp 0 (iii) quantifies how much of the action it could encode",
    )

    # -- in-frustum at every stored horizon ------------------------------
    if sim_main["obj_poses"] is not None:
        data = mujoco.MjData(sm.model)
        cam_id = mujoco.mj_name2id(sm.model, mujoco.mjtObj.mjOBJ_CAMERA, C.CAMERA_NAME)
        worst = -np.inf
        worst_h = None
        for hname, idx in horizon_idx.items():
            traj = sim_main["obj_poses"] if hname != "s_time" else (
                sim_time["obj_poses"] if sim_time is not None else sim_main["obj_poses"])
            if traj is None:
                continue
            data.qpos[:] = (sim_main if hname != "s_time" or sim_time is None
                            else sim_time)["qpos_traj"][int(idx)]
            data.qvel[:] = 0.0
            mujoco.mj_forward(sm.model, data)
            pts = _object_corner_points(geometry, traj[int(idx)])
            uv = _project(sm.model, cam_id, data, pts)
            if np.any(~np.isfinite(uv)):
                worst, worst_h = np.inf, hname
                break
            # distance outside the safe box (positive = outside)
            m = tol.frustum_margin_px
            excess = np.max(np.maximum(m - uv, uv - (C.RENDER_SIZE - m)))
            if excess > worst:
                worst, worst_h = float(excess), hname
        out["in_frustum"] = _v(
            "in_frustum", worst <= 0.0, worst, 0.0,
            worst_horizon=worst_h, margin_px=tol.frustum_margin_px,
            note="object must be fully inside the image at every stored horizon",
        )

    # -- non-settler ------------------------------------------------------
    # Not applicable to ABSENT: there is no object, so there is nothing to
    # settle.  Failing it there would exclude 100% of the OOD reference arm and
    # silently delete the T2 measurement.
    has_object = sim_main["obj_poses"] is not None
    out["settled"] = _v(
        "settled",
        (not has_object) or thresholds is None or settled_at is not None,
        -1.0 if settled_at is None else float(settled_at),
        None,
        applicable=has_object, thresholds_available=thresholds is not None,
        note="objects that never settle (sphere at low friction) are excluded and counted",
    )

    # -- T5: s_time control ------------------------------------------------
    if sim_time is not None:
        post = ph.push_end_idx + ph.eps_withdraw + 1
        cf = sim_time["contact_forces"]
        peak_post = float(np.max(cf[post:])) if cf[post:].size else 0.0
        out["T5_no_contact_after_withdraw"] = _v(
            "T5_no_contact_after_withdraw", peak_post <= tol.contact_force_zero_n,
            peak_post, tol.contact_force_zero_n,
            withdraw_complete_idx=int(post), eps_withdraw_m=C.EPS_WITHDRAW,
            note="eps-withdrawn arm must exert zero force on the settling object",
        )
        if sim_main["obj_poses"] is not None:
            idx = horizon_idx["s_del"]
            p1 = sim_main["obj_poses"][idx]
            p2 = sim_time["obj_poses"][idx]
            dpos, drot = D.pose_distance(p1, p2, geometry, C.GEOM_SIZE[geometry], quotient=True)
            out["T5_settled_pose_match_pos"] = _v(
                "T5_settled_pose_match_pos", dpos <= tol.settled_pose_pos_m,
                dpos, tol.settled_pose_pos_m,
                note="ground-truth settled pose must match between the s_del and s_time arms",
            )
            out["T5_settled_pose_match_rot"] = _v(
                "T5_settled_pose_match_rot", drot <= tol.settled_pose_rot_rad,
                drot, tol.settled_pose_rot_rad,
            )
        if "s_del" in seg_masks and "s_time" in seg_masks:
            n_del = float((seg_masks["s_del"] == C.SEG_OBJECT).sum())
            n_time = float((seg_masks["s_time"] == C.SEG_OBJECT).sum())
            denom = max(n_del, 1.0)
            rel = abs(n_del - n_time) / denom
            # LOGGED, NEVER ENFORCED -- and the reason matters.
            #
            # Some differential occlusion is inherent to the s_time control, not
            # a defect of it: the whole point of s_time is that the arm has NOT
            # retracted, so it necessarily sits beside the settled object and
            # can hide part of it, whereas at s_del it is at canonical rest.
            #
            # Excluding rollouts on this criterion would be actively harmful.
            # Occlusion is largest exactly when the object ends up close to
            # where the arm stopped, which is a function of the action -- so
            # excluding on it would carve an ACTION-CORRELATED hole in the
            # sample.  That is a far more dangerous bias than the occlusion it
            # would be removing.  It is therefore measured on every rollout,
            # aggregated by validate_corpus.py, and reported as a T5 caveat:
            # it inflates err(A-time) for a reason unrelated to retraction, and
            # so makes the T5 control conservative rather than optimistic.
            out["T5_occlusion_match"] = _v(
                "T5_occlusion_match", True, rel, tol.occlusion_frac_delta,
                object_px_s_del=n_del, object_px_s_time=n_time,
                enforced=False, exceeds_tolerance=bool(rel > tol.occlusion_frac_delta),
                note="DIAGNOSTIC ONLY: differential occlusion of the settled object "
                     "between the s_del and s_time arms; never an exclusion criterion, "
                     "because occlusion is action-correlated",
            )

    # -- s_0 / s_std must be shared between the two arms ------------------
    if sim_time is not None:
        pre = ph.push_end_idx + ph.eps_withdraw + 1
        d0 = float(np.max(np.abs(sim_main["arm_qpos"][:pre] - sim_time["arm_qpos"][:pre])))
        out["shared_prefix"] = _v(
            "shared_prefix", d0 <= 1e-9, d0, 1e-9,
            shared_until_idx=int(pre - 1),
            note="the s_del and s_time arms must be identical through s_std AND the "
                 "shared eps-withdrawal, i.e. until contact is provably broken",
        )

    out["_all_passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


def render_empty_scene(geometry: str = "box") -> np.ndarray:
    """The scene with neither arm nor object visible.

    Used as the fill for the §8.4 masking decomposition: masked-out regions are
    composited with this, not filled with black, so a probe cannot read the
    *shape* of the mask instead of the content it was meant to be denied.
    """
    sm = get_scene(geometry, "ABSENT", 1.0)
    data = mujoco.MjData(sm.model)
    mujoco.mj_resetData(sm.model, data)
    # Park the arm far outside the frustum rather than hiding it, for the same
    # reason ABSENT compiles away the object: a hidden body can still speckle.
    data.qpos[sm.arm_qadr] = [0.0, 0.0, 5.0, 0.0]
    mujoco.mj_forward(sm.model, data)
    with mujoco.Renderer(sm.model, height=C.RENDER_SIZE, width=C.RENDER_SIZE) as r:
        r.update_scene(data, camera=C.CAMERA_NAME)
        return r.render().copy()


def render_state(geometry: str, condition: str, arm_qpos, obj_pose=None,
                 *, segmentation: bool = False,
                 camera_offset=None) -> np.ndarray:
    """Render one arbitrary posed state.  Used by Experiment B's sweeps.

    ``camera_offset`` perturbs the camera position; everything else about the
    camera stays byte-identical, which is what makes the Experiment B noise
    floor a measurement of the *encoder* rather than of the renderer.
    """
    sm = get_scene(geometry, condition, 1.0)
    model = sm.model
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, C.CAMERA_NAME)
    saved = model.cam_pos[cam_id].copy()
    if camera_offset is not None:
        model.cam_pos[cam_id] = saved + np.asarray(camera_offset, dtype=np.float64)
    try:
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[sm.arm_qadr] = np.asarray(arm_qpos, dtype=np.float64)
        if sm.obj_qadr is not None:
            if obj_pose is None:
                raise ValueError("condition has an object but no pose was given")
            data.qpos[sm.obj_qadr : sm.obj_qadr + 7] = np.asarray(obj_pose, dtype=np.float64)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=C.RENDER_SIZE, width=C.RENDER_SIZE) as r:
            if segmentation:
                r.enable_segmentation_rendering()
            r.update_scene(data, camera=C.CAMERA_NAME)
            raw = r.render()
            return _seg_to_labels(sm, raw) if segmentation else raw.copy()
    finally:
        model.cam_pos[cam_id] = saved


def check_rest_pose_occlusion(geometry: str = "box", n_probe: int = 64) -> dict:
    """ASSERT (§6): the canonical rest pose does not occlude the settling region.

    Places the object at a grid of reachable settling positions with the arm at
    canonical rest, and measures how many object pixels the arm hides relative
    to a render with the arm moved far away.
    """
    sm = get_scene(geometry, "INTERACT", 1.0)
    rng = np.random.default_rng(0)
    # Must cover where objects actually come to rest, which at the lowest
    # Experiment F friction multiplier is well outside the swept corridor.
    reach = max_corridor_radius() + 0.16
    worst = 0.0
    worst_xy = None
    rest = np.array([C.REST_POSE[0], C.REST_POSE[1], C.REST_POSE[2], C.REST_YAW])
    far = np.array([0.0, 0.0, 0.65, 0.0])
    data = mujoco.MjData(sm.model)
    with mujoco.Renderer(sm.model, height=C.RENDER_SIZE, width=C.RENDER_SIZE) as seg_r:
        seg_r.enable_segmentation_rendering()

        def obj_px(arm_cfg, xy):
            mujoco.mj_resetData(sm.model, data)
            data.qpos[sm.arm_qadr] = arm_cfg
            data.qpos[sm.obj_qadr : sm.obj_qadr + 3] = [xy[0], xy[1], object_rest_height(geometry)]
            data.qpos[sm.obj_qadr + 3 : sm.obj_qadr + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(sm.model, data)
            seg_r.update_scene(data, camera=C.CAMERA_NAME)
            return float((_seg_to_labels(sm, seg_r.render()) == C.SEG_OBJECT).sum())

        for _ in range(n_probe):
            r = reach * math.sqrt(rng.uniform(0, 1))
            a = rng.uniform(-math.pi, math.pi)
            xy = (r * math.cos(a), r * math.sin(a))
            n_rest = obj_px(rest, xy)
            n_far = obj_px(far, xy)
            if n_far < 1:
                continue
            hidden = (n_far - n_rest) / n_far
            if hidden > worst:
                worst, worst_xy = float(hidden), xy

    return _v("rest_pose_no_occlusion", worst <= 0.01, worst, 0.01,
              worst_xy=None if worst_xy is None else list(worst_xy),
              probe_radius_m=reach, geometry=geometry,
              note="fraction of the object hidden by the arm at canonical rest")
