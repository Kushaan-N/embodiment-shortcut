"""Symmetry-quotiented state distances and prior baselines (§7.2, §7.3).

Quaternion convention is MuJoCo's: ``q = (w, x, y, z)``.

The two things this module exists to get right:

1. **Rotation distance is geodesic and double-cover-aware.**
   ``2 * arccos(|<q1, q2>|)``, never component-wise, because ``q ~= -q``.

2. **Rotation distance is quotiented by the object's proper symmetry group.**
   The group is *derived from the collision geometry's actual dimensions*, not
   assumed.  For the project's cube the derived group has order 24, matching
   §7.2; a box with unequal half-extents correctly yields order 8 or 4 instead.

Both quotiented and unquotiented variants are exposed everywhere, because §7.2
requires reporting both for the box and letting Experiment C adjudicate.
"""

from __future__ import annotations

import itertools
import math
from functools import lru_cache

import numpy as np

__all__ = [
    "quat_normalize", "quat_conjugate", "quat_multiply", "quat_to_matrix",
    "matrix_to_quat", "quat_from_axis_angle",
    "geodesic_quat_distance", "cube_rotation_group", "box_symmetry_group",
    "symmetry_group", "rotation_distance", "translation_distance",
    "pose_distance", "state_distance",
    "uniform_prior_mae", "uniform_prior_mse",
    "normalised_uniform_prior_mae", "normalised_uniform_prior_mse",
    "prior_baseline_table",
]

_EPS = 1e-12


# --------------------------------------------------------------------------
# Quaternion primitives
# --------------------------------------------------------------------------


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < _EPS):
        raise ValueError("zero-norm quaternion")
    return q / n


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, (w, x, y, z) convention, broadcasting over leading dims."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (w, x, y, z).  Shepperd's method (numerically stable)."""
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s]
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s]
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s]
    return quat_normalize(np.asarray(q))


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), _EPS)
    h = 0.5 * float(angle)
    return np.concatenate([[math.cos(h)], math.sin(h) * axis])


def geodesic_quat_distance(q1: np.ndarray, q2: np.ndarray) -> float | np.ndarray:
    """``2 * arccos(|<q1, q2>|)`` in radians, range [0, pi].

    The absolute value handles the double cover (q and -q are the same
    rotation).  Never compare quaternion components directly.
    """
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    out = 2.0 * np.arccos(dot)
    return float(out) if np.ndim(out) == 0 else out


# --------------------------------------------------------------------------
# Symmetry groups
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _signed_permutation_rotations() -> np.ndarray:
    """The 24 proper rotations of the cube, as 3x3 signed permutation matrices."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for row, col in enumerate(perm):
                M[row, col] = signs[row]
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                mats.append(M)
    assert len(mats) == 24, len(mats)
    return np.stack(mats)


@lru_cache(maxsize=1)
def cube_rotation_group() -> np.ndarray:
    """(24, 4) quaternions: the proper rotation group of a cube."""
    return np.stack([matrix_to_quat(M) for M in _signed_permutation_rotations()])


def box_symmetry_group(half_extents) -> np.ndarray:
    """Proper rotation symmetry group of an axis-aligned box, derived from its size.

    A signed permutation ``R`` is a symmetry iff it maps the half-extent vector
    to itself, i.e. ``|R| h == h``.  This yields |G| = 24 for a cube, 8 for a
    square prism, 4 for a fully generic box -- automatically and correctly.
    """
    h = np.asarray(half_extents, dtype=np.float64).reshape(3)
    keep = []
    for M, q in zip(_signed_permutation_rotations(), cube_rotation_group()):
        if np.allclose(np.abs(M) @ h, h, atol=1e-12, rtol=0.0):
            keep.append(q)
    G = np.stack(keep)
    assert len(G) in (4, 8, 12, 24), f"unexpected group order {len(G)} for {h}"
    return G


_IDENTITY_GROUP = np.array([[1.0, 0.0, 0.0, 0.0]])


def symmetry_group(geometry: str, size=None) -> np.ndarray:
    """Discrete proper symmetry group used for quotienting.

    ``sphere`` and ``cylinder`` have *continuous* symmetry; they are handled
    analytically in ``rotation_distance`` rather than by a finite group, and
    this function returns the trivial group for them.
    """
    if geometry == "box":
        if size is None:
            raise ValueError("box symmetry group needs the half-extents")
        return box_symmetry_group(size)
    if geometry in ("sphere", "cylinder"):
        return _IDENTITY_GROUP
    raise ValueError(f"unknown geometry {geometry!r}")


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------


def translation_distance(p1, p2) -> float | np.ndarray:
    d = np.asarray(p1, dtype=np.float64) - np.asarray(p2, dtype=np.float64)
    out = np.linalg.norm(d, axis=-1)
    return float(out) if np.ndim(out) == 0 else out


def _body_axis(q, axis_index: int = 2) -> np.ndarray:
    R = quat_to_matrix(q)
    return R[..., :, axis_index]


def rotation_distance(q1, q2, geometry: str, size=None, quotient: bool = True) -> float:
    """Rotation distance in radians, per §7.2.

    quotient=True:
      * ``box``      -> min over the box's proper rotation group (24 for a cube)
      * ``sphere``   -> 0 (rotation excluded entirely)
      * ``cylinder`` -> axial rotation excluded: ``arccos(|<z1, z2>|)``, the
                        angle between the symmetry axes, with the absolute value
                        because a uniform cylinder is also symmetric under a
                        180-degree flip.  Range [0, pi/2].
    quotient=False: plain geodesic for every geometry.
    """
    q1 = quat_normalize(np.asarray(q1, dtype=np.float64))
    q2 = quat_normalize(np.asarray(q2, dtype=np.float64))

    if not quotient:
        return float(geodesic_quat_distance(q1, q2))

    if geometry == "sphere":
        return 0.0

    if geometry == "cylinder":
        z1 = _body_axis(q1)
        z2 = _body_axis(q2)
        c = float(np.clip(abs(float(np.dot(z1, z2))), -1.0, 1.0))
        return float(np.arccos(c))

    if geometry == "box":
        G = symmetry_group("box", size)
        # q2 composed with each body-frame symmetry is the same physical pose.
        cands = quat_multiply(np.broadcast_to(q2, G.shape), G)
        return float(np.min(geodesic_quat_distance(np.broadcast_to(q1, cands.shape), cands)))

    raise ValueError(f"unknown geometry {geometry!r}")


def pose_distance(pose1, pose2, geometry: str, size=None, quotient: bool = True):
    """Split (translation_m, rotation_rad) for two 7-vectors ``(x, y, z, w, qx, qy, qz)``."""
    pose1 = np.asarray(pose1, dtype=np.float64).reshape(7)
    pose2 = np.asarray(pose2, dtype=np.float64).reshape(7)
    dpos = translation_distance(pose1[:3], pose2[:3])
    drot = rotation_distance(pose1[3:], pose2[3:], geometry, size, quotient=quotient)
    return float(dpos), float(drot)


def state_distance(
    pose1,
    pose2,
    geometry: str,
    size=None,
    *,
    pos_scale: float,
    rot_scale: float,
    quotient: bool = True,
) -> float:
    """Scalar state distance with per-component normalisation (§9-G).

    ``pos_scale`` / ``rot_scale`` are the Experiment-A delta scales for this
    geometry.  Mixing raw metres with raw radians (or Newtons) is forbidden.
    """
    if pos_scale <= 0 or rot_scale <= 0:
        raise ValueError("scales must be positive; pass Experiment A's delta scales")
    dpos, drot = pose_distance(pose1, pose2, geometry, size, quotient=quotient)
    return float(math.hypot(dpos / pos_scale, drot / rot_scale))


# --------------------------------------------------------------------------
# Prior baselines (§7.3)
# --------------------------------------------------------------------------


def uniform_prior_mae(lo: float, hi: float) -> float:
    """MAE of the best constant predictor (the median = midpoint) on U[lo, hi]."""
    return (hi - lo) / 4.0


def uniform_prior_mse(lo: float, hi: float) -> float:
    """MSE of the best constant predictor (the mean = midpoint) on U[lo, hi]."""
    return (hi - lo) ** 2 / 12.0


def normalised_uniform_prior_mae(lo: float, hi: float) -> float:
    """MAE normalised by the sampling range: exactly 1/4 for any uniform."""
    return uniform_prior_mae(lo, hi) / (hi - lo)


def normalised_uniform_prior_mse(lo: float, hi: float) -> float:
    """MSE normalised by range^2: exactly 1/12 for any uniform."""
    return uniform_prior_mse(lo, hi) / (hi - lo) ** 2


def prior_baseline_table(action_ranges: dict) -> dict:
    """Analytic prior baseline per action dimension, plus the pooled mean.

    ``analyze.py`` cross-checks these against the empirical constant-predictor
    fit on train; the two agreeing is itself a sanity check (§7.3).
    """
    per_dim = {}
    for name, (lo, hi) in action_ranges.items():
        per_dim[name] = {
            "range": [lo, hi],
            "mae": uniform_prior_mae(lo, hi),
            "mse": uniform_prior_mse(lo, hi),
            "mae_norm": normalised_uniform_prior_mae(lo, hi),
            "mse_norm": normalised_uniform_prior_mse(lo, hi),
        }
    return {
        "per_dim": per_dim,
        "pooled_mae_norm": float(np.mean([v["mae_norm"] for v in per_dim.values()])),
        "pooled_mse_norm": float(np.mean([v["mse_norm"] for v in per_dim.values()])),
    }
