"""Global configuration for the embodiment-shortcut / OG-AF project.

Everything that a downstream number depends on lives here or in
``thresholds.json`` (written by ``exp_a_deltas.py``).  Nothing physical is
hardcoded at a call site.

Two hard rules encoded in this module:

* **Thresholds are never hardcoded.**  ``load_thresholds()`` raises if
  ``thresholds.json`` is missing; it is produced by Experiment A (§9-A) and is
  the sole source of ``delta_pos_min``, ``delta_rot_min`` and the settling
  speed cutoff (§6.4).
* **Action dimensions are non-circular bounded uniforms.**  The whole
  prior-baseline anchor of §7.3 assumes this (the midpoint of a circular
  variable is meaningless).  ``ASSERT_ACTION_NONCIRCULAR`` is checked at import.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("OGAF_DATA", REPO_ROOT / "data"))
RESULTS_ROOT = Path(os.environ.get("OGAF_RESULTS", REPO_ROOT / "results"))
CORPUS_ROOT = Path(os.environ.get("OGAF_CORPUS", DATA_ROOT / "corpus"))
CLIP_ROOT = Path(os.environ.get("OGAF_CLIPS", DATA_ROOT / "clips"))
CHECKPOINT_ROOT = Path(os.environ.get("OGAF_CKPT", DATA_ROOT / "checkpoints"))
THRESHOLDS_PATH = Path(os.environ.get("OGAF_THRESHOLDS", RESULTS_ROOT / "exp_a" / "thresholds.json"))

for _p in (DATA_ROOT, RESULTS_ROOT, CORPUS_ROOT, CHECKPOINT_ROOT):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Action space  (§6:  a in R^3 = push velocity, planar approach angle, contact offset)
# --------------------------------------------------------------------------

ACTION_DIMS: tuple[str, ...] = ("push_speed", "approach_angle", "contact_offset")

#: Uniform sampling ranges.  These *define* the prior baseline of §7.3.
ACTION_RANGES: dict[str, tuple[float, float]] = {
    "push_speed": (0.12, 0.36),                    # m/s, commanded pusher speed
    "approach_angle": (-math.pi / 2, math.pi / 2),  # rad, relative to +x base direction
    "contact_offset": (-0.035, 0.035),              # m, lateral offset at the nominal target
}

#: §7.3 requires a well-defined constant-predictor baseline; a wrapped/circular
#: dimension has no midpoint and would make the anchor meaningless.  The
#: approach angle is therefore restricted to a non-wrapping half-turn arc.
ASSERT_ACTION_NONCIRCULAR = True

BASE_APPROACH_DIR = (1.0, 0.0)  # +x; approach_angle rotates this in the ground plane


def action_lo_hi() -> tuple["list[float]", "list[float]"]:
    lo = [ACTION_RANGES[d][0] for d in ACTION_DIMS]
    hi = [ACTION_RANGES[d][1] for d in ACTION_DIMS]
    return lo, hi


def action_ranges_array():
    import numpy as np

    lo, hi = action_lo_hi()
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

GEOMETRIES: tuple[str, ...] = ("box", "sphere", "cylinder")

#: Half-extents / radii, in metres.  The box is a *cube* (all half-extents
#: equal), so its proper rotation symmetry group has order 24 as §7.2 states.
#: distances.box_symmetry_group() derives the group from these numbers rather
#: than assuming 24, so a future non-cube box stays correct.
GEOM_SIZE: dict[str, tuple[float, ...]] = {
    "box": (0.040, 0.040, 0.040),   # half-extents -> cube -> |group| = 24
    "sphere": (0.040,),             # radius
    "cylinder": (0.040, 0.040),     # radius, half-height
}

OBJECT_DENSITY = 500.0  # kg/m^3

#: Friction of the *object* geom, per geometry.  The object geom carries
#: ``priority=1`` so these values (not the plane's) govern every object
#: contact, which is what makes "object friction" unambiguous for Experiment F.
#:
#: The sphere needs its own rolling-friction coefficient.  §14 anticipates this
#: ("sphere may never settle at low friction ... per-geometry settling
#: policies"): with the box's rolling friction a pushed sphere rolls ~0.2 m on
#: average and leaves the camera frustum in ~30% of rollouts, so the geometry
#: would contribute almost nothing but exclusions.  Sliding friction is held
#: identical across geometries so that Experiment F's dose-response sweep --
#: which scales the sliding component only -- stays comparable.
OBJECT_FRICTION_BY_GEOMETRY: dict[str, tuple[float, float, float]] = {
    "box": (0.35, 0.005, 0.0005),       # (sliding, torsional, rolling)
    "sphere": (0.35, 0.005, 0.0200),
    "cylinder": (0.35, 0.005, 0.0005),
}
OBJECT_FRICTION = OBJECT_FRICTION_BY_GEOMETRY["box"]  # nominal, for reporting
PLANE_FRICTION = (0.35, 0.005, 0.0001)

#: Experiment F dose-response multipliers applied to the *sliding* component
#: only (torsional/rolling held fixed) -- log-spaced, 5 points (§9-F).
FRICTION_MULTIPLIERS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)

# --------------------------------------------------------------------------
# Arm / pusher
# --------------------------------------------------------------------------

#: "kinematic": arm qpos/qvel are overwritten with the commanded values before
#:   every mj_step -> quasi-infinite impedance, the T9 escalation the spec
#:   permits ("a mocap-welded kinematic body").  Contact impulses on the object
#:   are still computed from the true relative state.
#: "position": high-gain PD position actuators.  Realistic, but the T9
#:   deviation assertion may fail; the validator measures and reports it either
#:   way.  Switch with OGAF_ARM_CONTROL=position.
ARM_CONTROL = os.environ.get("OGAF_ARM_CONTROL", "kinematic")
assert ARM_CONTROL in ("kinematic", "position")

ARM_KP = 40000.0          # position-actuator gain when ARM_CONTROL == "position"
ARM_DAMPING = 400.0       # joint damping

#: The pusher is a 4-DOF gantry: slide x, slide y, slide z, hinge yaw.
#:
#: DEVIATION FROM SPEC §6, stated deliberately.  §6 asks for "a 2-3 DOF pusher
#: arm".  With 3 DOF the arm's *visible configuration* at a single horizon is
#: (x, y) at a constant push height -- two numbers -- which cannot be injective
#: in a three-dimensional action.  Claim C0 is precisely the statement that the
#: arm state at s_std recovers the action to near the identifiability floor, so
#: a 3-DOF arm would refute C0 for a trivial dimension-counting reason rather
#: than a scientific one, and would make the whole shortcut argument untestable.
#: The yaw joint makes the map (v, theta, d) -> (x, y, yaw) analytically
#: invertible, which is exactly the "actuator command is a near-direct readout
#: of manipulator pose" situation the paper is about.  Recorded in
#: metadata.json as `arm_dof_deviation`.
ARM_JOINTS: tuple[str, ...] = ("arm_x", "arm_y", "arm_z", "arm_yaw")
ARM_DOF_DEVIATION = (
    "4-DOF (x, y, z, yaw) instead of spec §6's 2-3 DOF: a 3-DOF arm's "
    "single-horizon visible state is 2-dimensional and cannot be injective in "
    "a 3-dimensional action, which would refute C0 by dimension counting."
)

#: Thin paddle so that yaw is legible in a 224x224 render.
PADDLE_HALF_EXTENTS = (0.005, 0.028, 0.060)  # (thickness, half-width, half-height)
STALK_RADIUS = 0.012
STALK_TOP_REL = 0.09      # stalk runs from paddle top to this height above centre

PUSH_HEIGHT = 0.065       # m, paddle centre z during the push (bottom at 0.005)

REST_POSE = (-0.10, -0.10, 0.25)  # canonical rest (x, y, z) of the paddle centre
REST_YAW = 0.0                   # canonical rest yaw
PUSH_START_RADIUS = 0.070        # m from the nominal target, along -approach_dir

# --------------------------------------------------------------------------
# Timing / phases (§6.4).  All durations in seconds; converted to steps.
# --------------------------------------------------------------------------

SIM_DT = 0.002

#: Nominal video frame rate of the world models this metric is used to score.
#:
#: This matters more than it looks.  Experiment A's ``delta_pos_min`` is the
#: smallest object displacement that the metric could ever be asked to notice,
#: and Experiment B gates the whole project on whether the encoder can resolve
#: it.  Measured at the 500 Hz *physics* timestep, that delta is a fraction of
#: a micron and Gate B would fail as an artefact of the integrator step rather
#: than as a fact about DINO.  The physically meaningful scale is the
#: displacement between consecutive *video* frames, so Experiment A measures
#: deltas at FRAME_STRIDE and clips are sampled on the same grid.
VIDEO_FPS = 20.0
FRAME_STRIDE = int(round(1.0 / (VIDEO_FPS * SIM_DT)))  # physics steps per video frame

T_HOLD0 = 0.10     # settle the scene, then render s_0
T_TRAVERSE = 0.40  # rest -> above push start, at rest height (clears the object)
T_DESCEND = 0.15   # descend to push height at the push-start xy
T_PUSH = 0.30      # constant-velocity push; s_std is the last step of this phase
T_RETREAT = 0.15   # back off along -approach_dir before lifting
T_LIFT_HOME = 0.40 # return to canonical rest
T_SETTLE_MAX = 2.0 # cap; non-settlers are marked and excluded (§6.4)

#: s_time arm: withdraw by eps along -approach_dir, then hold (T5 v3 fix).
EPS_WITHDRAW = 0.0015   # m -- just clears the contact margin
T_EPS_WITHDRAW = 0.02   # s
#: Distance the retracting arm backs off along -u before lifting.  Kept small
#: on purpose: the retreat is only there to break contact before the lift, and
#: it is what sets the outer radius of the swept corridor at *low* push speeds
#: (where push_end is already behind the nominal target).  A generous retreat
#: forces the DECOY annulus outward, which forces the camera back, which shrinks
#: the object in pixels and makes Gate B harder for no scientific reason.
RETREAT_DISTANCE = 0.030  # m, for the retracting arm

# --------------------------------------------------------------------------
# Conditions (§6.3)
# --------------------------------------------------------------------------

CONDITIONS: tuple[str, ...] = ("INTERACT", "DECOY", "ABSENT")

#: DECOY placement annulus, in metres from the nominal target.  Chosen so that
#: it is disjoint from the union of *every* reachable push corridor, which lets
#: the placement be sampled with no action-dependent rejection at all -- the
#: cleanest possible defence against T11.  ``scene.max_corridor_radius()``
#: asserts the disjointness numerically.
DECOY_ANNULUS = (0.19, 0.21)

#: Independent per-rollout jitter of the INTERACT object's initial pose, so the
#: task is not degenerate.  Sampled from an RNG stream *disjoint* from the
#: action stream (T11).
INTERACT_POS_JITTER = 0.005   # m, uniform in a square
INTERACT_YAW_JITTER = True    # uniform in [0, 2pi)

# --------------------------------------------------------------------------
# Rendering (§6)
# --------------------------------------------------------------------------

RENDER_SIZE = 224  # DINO native; no resize step anywhere
CAMERA_NAME = "fixed_cam"
CAMERA_POS = (0.375, -0.375, 0.57)
CAMERA_TARGET = (0.0, 0.0, 0.04)
CAMERA_FOVY = 45.0

#: Segmentation label ids used everywhere (§8.4 masking decomposition).
SEG_BACKGROUND = 0
SEG_ARM = 1
SEG_OBJECT = 2

# --------------------------------------------------------------------------
# Corpus / splits (§8.2, T7)
# --------------------------------------------------------------------------

N_TUPLES_PER_GEOMETRY = 2000   # (action, seed) tuples; x3 conditions each
SPLIT_FRACTIONS = (0.80, 0.10, 0.10)
SPLIT_UNIT = "action_seed_tuple"  # NEVER "frame", NEVER "rollout" (T7)
SPLIT_SALT = "ogaf-v3-split"      # fixed; changing it invalidates every split

MASTER_SEED = 20260817
SHARD_SIZE = 100  # rollout tuples per shard file (Modal volumes race across containers)

# --------------------------------------------------------------------------
# IDM training (§8)
# --------------------------------------------------------------------------

ARCH_A_SEEDS = tuple(range(5))    # 5 seeds
ARCH_B_SEEDS = tuple(range(10))   # 10 seeds -- probes are cheap
MASKING_SEEDS = tuple(range(5))   # 5 seeds

#: Variant -> horizon key it reads as the second frame.
IDM_VARIANTS: dict[str, str] = {
    "A-std": "s_std",
    "A-del": "s_del",
    "A-time": "s_time",
    "A-clip-del": "s_del",
    "B-std": "s_std",
    "B-del": "s_del",
    "B-time": "s_time",
}

#: Which variants consume a clip rather than a frame pair (§8.1, T8).
#: MultiWorld §B.2 trains a *bidirectional* IDM "following VPT", so the standard
#: metric's input regime is a clip, not a pair.  See protocol/multiworld_b2.yaml.
CLIP_VARIANTS: tuple[str, ...] = ("A-std", "A-clip-del")

#: Number of frames per clip.  Not stated by MultiWorld §B.2; see the protocol
#: file for the provenance record and justification.
CLIP_LENGTH = 16

ENCODERS = ("facebook/dinov2-large", "facebook/dinov3-vitl16-pretrain-lvd1689m")
ENCODER_POOLINGS = ("cls", "mean_patch")

# --------------------------------------------------------------------------
# Statistics (§10, pre-registered)
# --------------------------------------------------------------------------

N_BOOTSTRAP = 10_000
BOOTSTRAP_CI = 0.95
BOOTSTRAP_SEED = 987654321
TEST_SIDEDNESS = "one-sided"
TAU_1_DEFAULT = 0.25   # C1 threshold as a fraction of the s_std floor (§11)
POWER_ALPHA = 0.05
POWER_TARGET = 0.90
POWER_MDE_FRACTION_OF_FLOOR = 0.25

# --------------------------------------------------------------------------
# Validator tolerances
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerances:
    """Tolerances for the §6 validators.

    ``arm_deviation_m`` is the T9 tolerance.  Its *target* is the Experiment B
    resolution floor -- i.e. the encoder provably cannot see the residual.
    Until Experiment B has run, the fallback below is used and the validator
    records which of the two it compared against.
    """

    arm_deviation_m: float = 5e-4
    settled_pose_pos_m: float = 2e-3     # s_del vs s_time settled-position match (T5)
    settled_pose_rot_rad: float = 0.05   # s_del vs s_time settled-rotation match (T5)
    contact_force_zero_n: float = 1e-6   # "zero" contact force (DECOY, post-withdraw)
    frustum_margin_px: float = 4.0       # object must be this far inside the image
    decoy_placement_r2_max: float = 0.01 # T11 ridge-regression gate
    occlusion_frac_delta: float = 0.05   # s_del vs s_time object-visibility mismatch


TOLERANCES = Tolerances()


# --------------------------------------------------------------------------
# Thresholds produced by Experiment A -- never hardcoded (§9-A)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """Physical scales derived from Experiment A."""

    delta_pos_min: dict          # geometry -> m (5th pct of nonzero active-phase deltas)
    delta_rot_min: dict          # geometry -> rad, symmetry-quotiented (None if excluded)
    delta_rot_raw_min: dict      # geometry -> rad, raw geodesic (always defined)
    settle_lin_speed: dict       # geometry -> m/s
    settle_ang_speed: dict       # geometry -> rad/s
    pooled_delta_pos_min: float
    pooled_delta_rot_min: float
    source_run: str

    def for_geometry(self, geometry: str) -> dict:
        return {
            "delta_pos_min": self.delta_pos_min[geometry],
            "delta_rot_min": self.delta_rot_min[geometry],
            "delta_rot_raw_min": self.delta_rot_raw_min[geometry],
            "settle_lin_speed": self.settle_lin_speed[geometry],
            "settle_ang_speed": self.settle_ang_speed[geometry],
        }


class ThresholdsMissing(RuntimeError):
    pass


def load_thresholds(path: Path | None = None) -> Thresholds:
    """Load Experiment A's thresholds, or raise.

    There is deliberately no default: §9-A says "never hardcode 1mm or 1deg".
    """
    p = Path(path) if path is not None else THRESHOLDS_PATH
    if not p.exists():
        raise ThresholdsMissing(
            f"{p} not found.  Run `python exp_a_deltas.py` first -- settling and "
            f"resolution thresholds are derived from measurement, never hardcoded (§9-A)."
        )
    with open(p) as fh:
        raw = json.load(fh)
    return Thresholds(
        delta_pos_min=raw["delta_pos_min"],
        delta_rot_min=raw["delta_rot_min"],
        delta_rot_raw_min=raw["delta_rot_raw_min"],
        settle_lin_speed=raw["settle_lin_speed"],
        settle_ang_speed=raw["settle_ang_speed"],
        pooled_delta_pos_min=raw["pooled_delta_pos_min"],
        pooled_delta_rot_min=raw["pooled_delta_rot_min"],
        source_run=raw.get("source_run", "unknown"),
    )


# --------------------------------------------------------------------------
# Derived step counts
# --------------------------------------------------------------------------


def _steps(seconds: float) -> int:
    return int(round(seconds / SIM_DT))


@dataclass(frozen=True)
class PhaseSteps:
    hold0: int
    traverse: int
    descend: int
    push: int
    retreat: int
    lift_home: int
    settle_max: int
    eps_withdraw: int

    @property
    def s0_idx(self) -> int:
        return 0

    @property
    def push_start_idx(self) -> int:
        return self.hold0 + self.traverse + self.descend

    @property
    def push_end_idx(self) -> int:
        """Index of s_std: the last step of the constant-velocity push."""
        return self.push_start_idx + self.push - 1

    @property
    def arm_rest_idx(self) -> int:
        """First step at which the retracting arm is back at canonical rest."""
        return self.push_end_idx + self.retreat + self.lift_home

    @property
    def max_steps(self) -> int:
        return self.arm_rest_idx + self.settle_max + 1


PHASES = PhaseSteps(
    hold0=_steps(T_HOLD0),
    traverse=_steps(T_TRAVERSE),
    descend=_steps(T_DESCEND),
    push=_steps(T_PUSH),
    retreat=_steps(T_RETREAT),
    lift_home=_steps(T_LIFT_HOME),
    settle_max=_steps(T_SETTLE_MAX),
    eps_withdraw=_steps(T_EPS_WITHDRAW),
)


# --------------------------------------------------------------------------
# Import-time invariants
# --------------------------------------------------------------------------


def _check_noncircular_actions() -> None:
    if not ASSERT_ACTION_NONCIRCULAR:
        return
    lo, hi = ACTION_RANGES["approach_angle"]
    span = hi - lo
    assert span < 2 * math.pi - 1e-9, (
        "approach_angle spans a full turn; it is then circular and §7.3's "
        "constant-predictor prior baseline is undefined.  Restrict the range."
    )
    for name, (a, b) in ACTION_RANGES.items():
        assert b > a, f"empty range for action dim {name}: ({a}, {b})"


_check_noncircular_actions()


def as_dict() -> dict:
    """Everything a run should record in metadata.json."""
    return {
        "action_dims": list(ACTION_DIMS),
        "action_ranges": {k: list(v) for k, v in ACTION_RANGES.items()},
        "base_approach_dir": list(BASE_APPROACH_DIR),
        "geometries": list(GEOMETRIES),
        "geom_size": {k: list(v) for k, v in GEOM_SIZE.items()},
        "object_density": OBJECT_DENSITY,
        "object_friction_by_geometry": {k: list(v) for k, v in
                                        OBJECT_FRICTION_BY_GEOMETRY.items()},
        "plane_friction": list(PLANE_FRICTION),
        "friction_multipliers": list(FRICTION_MULTIPLIERS),
        "arm_control": ARM_CONTROL,
        "arm_kp": ARM_KP,
        "arm_damping": ARM_DAMPING,
        "arm_joints": list(ARM_JOINTS),
        "arm_dof_deviation": ARM_DOF_DEVIATION,
        "paddle_half_extents": list(PADDLE_HALF_EXTENTS),
        "stalk_radius": STALK_RADIUS,
        "push_height": PUSH_HEIGHT,
        "rest_pose": list(REST_POSE),
        "rest_yaw": REST_YAW,
        "push_start_radius": PUSH_START_RADIUS,
        "sim_dt": SIM_DT,
        "video_fps": VIDEO_FPS,
        "frame_stride": FRAME_STRIDE,
        "phase_seconds": {
            "hold0": T_HOLD0, "traverse": T_TRAVERSE, "descend": T_DESCEND,
            "push": T_PUSH, "retreat": T_RETREAT, "lift_home": T_LIFT_HOME,
            "settle_max": T_SETTLE_MAX, "eps_withdraw": T_EPS_WITHDRAW,
        },
        "phase_steps": asdict(PHASES),
        "eps_withdraw_m": EPS_WITHDRAW,
        "retreat_distance_m": RETREAT_DISTANCE,
        "conditions": list(CONDITIONS),
        "decoy_annulus": list(DECOY_ANNULUS),
        "interact_pos_jitter": INTERACT_POS_JITTER,
        "interact_yaw_jitter": INTERACT_YAW_JITTER,
        "render_size": RENDER_SIZE,
        "camera": {"pos": list(CAMERA_POS), "target": list(CAMERA_TARGET), "fovy": CAMERA_FOVY},
        "n_tuples_per_geometry": N_TUPLES_PER_GEOMETRY,
        "split_fractions": list(SPLIT_FRACTIONS),
        "split_unit": SPLIT_UNIT,
        "split_salt": SPLIT_SALT,
        "master_seed": MASTER_SEED,
        "arch_a_seeds": list(ARCH_A_SEEDS),
        "arch_b_seeds": list(ARCH_B_SEEDS),
        "masking_seeds": list(MASKING_SEEDS),
        "idm_variants": dict(IDM_VARIANTS),
        "clip_variants": list(CLIP_VARIANTS),
        "clip_length": CLIP_LENGTH,
        "encoders": list(ENCODERS),
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_ci": BOOTSTRAP_CI,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "tau_1_default": TAU_1_DEFAULT,
        "tolerances": asdict(TOLERANCES),
    }
