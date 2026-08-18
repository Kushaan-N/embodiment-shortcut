"""Unit tests for §7.2 distances, against hand-computed cases.

The box symmetry group is the part most likely to be silently wrong, so it is
tested three ways: order, group closure, and specific rotations whose
quotiented distance is known by hand.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import distances as D  # noqa: E402

I = np.array([1.0, 0.0, 0.0, 0.0])


def rx(a):
    return D.quat_from_axis_angle([1, 0, 0], a)


def ry(a):
    return D.quat_from_axis_angle([0, 1, 0], a)


def rz(a):
    return D.quat_from_axis_angle([0, 0, 1], a)


# ---------------------------------------------------------------- geodesic


def test_geodesic_identity_is_zero():
    assert D.geodesic_quat_distance(I, I) == pytest.approx(0.0, abs=1e-12)


def test_geodesic_double_cover():
    """q and -q are the same rotation; a component-wise distance would say 2."""
    q = rz(1.234)
    assert D.geodesic_quat_distance(q, -q) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("angle", [0.1, 0.5, 1.0, 2.0, 3.0, math.pi])
def test_geodesic_equals_rotation_angle(angle):
    assert D.geodesic_quat_distance(I, rz(angle)) == pytest.approx(angle, abs=1e-9)


def test_geodesic_max_is_pi():
    assert D.geodesic_quat_distance(I, rz(math.pi)) == pytest.approx(math.pi, abs=1e-9)


def test_geodesic_is_symmetric_and_bi_invariant():
    a, b, g = rz(0.7), rx(1.1), ry(0.3)
    d = D.geodesic_quat_distance(a, b)
    assert D.geodesic_quat_distance(b, a) == pytest.approx(d, abs=1e-12)
    assert D.geodesic_quat_distance(
        D.quat_multiply(g, a), D.quat_multiply(g, b)
    ) == pytest.approx(d, abs=1e-9)


# ---------------------------------------------------- symmetry group orders


def test_cube_group_has_order_24():
    assert D.cube_rotation_group().shape == (24, 4)


def test_cube_group_elements_are_unique_rotations():
    G = D.cube_rotation_group()
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            assert D.geodesic_quat_distance(G[i], G[j]) > 1e-6


def test_cube_group_is_closed_under_multiplication():
    """A set that is not closed is not a group and the quotient is not a metric."""
    G = D.cube_rotation_group()
    for i in range(len(G)):
        for j in range(len(G)):
            prod = D.quat_multiply(G[i], G[j])
            assert min(D.geodesic_quat_distance(prod, g) for g in G) < 1e-9


@pytest.mark.parametrize(
    "half_extents,order",
    [
        ((0.04, 0.04, 0.04), 24),   # cube
        ((0.04, 0.04, 0.06), 8),    # square prism -> D4
        ((0.02, 0.04, 0.06), 4),    # generic cuboid -> Klein four-group
    ],
)
def test_box_symmetry_group_order_is_derived_from_size(half_extents, order):
    assert len(D.box_symmetry_group(half_extents)) == order


def test_project_box_is_a_cube_so_order_is_24():
    """§7.2 says 24; that is only true because config's box has equal extents."""
    import config as C

    assert len(D.symmetry_group("box", C.GEOM_SIZE["box"])) == 24


# ------------------------------------------------------- quotiented distance

CUBE = (0.04, 0.04, 0.04)


@pytest.mark.parametrize("q", [rz(math.pi / 2), rx(math.pi / 2), ry(math.pi),
                               D.quat_multiply(rz(math.pi / 2), rx(math.pi / 2))])
def test_cube_symmetries_have_zero_quotiented_distance(q):
    assert D.rotation_distance(I, q, "box", CUBE, quotient=True) == pytest.approx(0.0, abs=1e-9)


def test_cube_unquotiented_distance_is_not_zero():
    assert D.rotation_distance(I, rz(math.pi / 2), "box", CUBE, quotient=False) == \
        pytest.approx(math.pi / 2, abs=1e-9)


def test_cube_quotiented_distance_of_small_rotation_is_the_rotation():
    a = 0.2
    assert D.rotation_distance(I, rz(a), "box", CUBE, quotient=True) == pytest.approx(a, abs=1e-9)


def test_cube_quotiented_distance_is_at_most_60_degrees():
    """Max distance to the nearest of 24 cube rotations is the covering radius."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        v = rng.normal(size=4)
        q = D.quat_normalize(v)
        worst = max(worst, D.rotation_distance(I, q, "box", CUBE, quotient=True))
    assert worst < math.radians(63.0)


def test_generic_box_180_is_symmetry_but_90_is_not():
    hx = (0.02, 0.04, 0.06)
    assert D.rotation_distance(I, rz(math.pi), "box", hx, quotient=True) == \
        pytest.approx(0.0, abs=1e-9)
    assert D.rotation_distance(I, rz(math.pi / 2), "box", hx, quotient=True) == \
        pytest.approx(math.pi / 2, abs=1e-9)


# ------------------------------------------------------------------ sphere


@pytest.mark.parametrize("q", [rz(0.4), rx(2.0), ry(math.pi)])
def test_sphere_rotation_is_excluded_entirely(q):
    assert D.rotation_distance(I, q, "sphere", (0.04,), quotient=True) == 0.0


def test_sphere_unquotiented_still_measures_rotation():
    assert D.rotation_distance(I, rz(1.0), "sphere", (0.04,), quotient=False) == \
        pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------- cylinder

CYL = (0.04, 0.04)


@pytest.mark.parametrize("a", [0.3, 1.0, math.pi / 2, math.pi])
def test_cylinder_axial_rotation_is_excluded(a):
    assert D.rotation_distance(I, rz(a), "cylinder", CYL, quotient=True) == \
        pytest.approx(0.0, abs=1e-9)


def test_cylinder_tipping_is_measured():
    assert D.rotation_distance(I, rx(math.pi / 2), "cylinder", CYL, quotient=True) == \
        pytest.approx(math.pi / 2, abs=1e-9)


def test_cylinder_flip_is_a_symmetry():
    """A uniform cylinder is symmetric under a 180-degree transverse flip."""
    assert D.rotation_distance(I, rx(math.pi), "cylinder", CYL, quotient=True) == \
        pytest.approx(0.0, abs=1e-9)


def test_cylinder_quotiented_distance_never_exceeds_half_pi():
    rng = np.random.default_rng(1)
    for _ in range(300):
        q = D.quat_normalize(rng.normal(size=4))
        assert D.rotation_distance(I, q, "cylinder", CYL, quotient=True) <= math.pi / 2 + 1e-9


# --------------------------------------------------------- quotient <= raw


def test_quotienting_never_increases_distance():
    rng = np.random.default_rng(2)
    for geom, size in [("box", CUBE), ("sphere", (0.04,)), ("cylinder", CYL)]:
        for _ in range(200):
            q1 = D.quat_normalize(rng.normal(size=4))
            q2 = D.quat_normalize(rng.normal(size=4))
            dq = D.rotation_distance(q1, q2, geom, size, quotient=True)
            dr = D.rotation_distance(q1, q2, geom, size, quotient=False)
            assert dq <= dr + 1e-9


# ------------------------------------------------------------ matrix round-trip


def test_matrix_quat_roundtrip():
    rng = np.random.default_rng(3)
    for _ in range(200):
        q = D.quat_normalize(rng.normal(size=4))
        q2 = D.matrix_to_quat(D.quat_to_matrix(q))
        # 2*arccos(1-eps) ~ 2*sqrt(2*eps): a 1e-16 float64 error in the dot
        # product becomes ~3e-8 in the distance.  That is the metric's
        # conditioning near zero, not a roundtrip error.
        assert D.geodesic_quat_distance(q, q2) == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------- misc


def test_translation_distance():
    assert D.translation_distance([0, 0, 0], [3, 4, 0]) == pytest.approx(5.0)


def test_state_distance_requires_positive_scales():
    p = np.concatenate([[0, 0, 0], I])
    with pytest.raises(ValueError):
        D.state_distance(p, p, "box", CUBE, pos_scale=0.0, rot_scale=1.0)


def test_state_distance_is_normalised_not_raw_units():
    """Mixing raw metres with raw radians is forbidden (§9-G)."""
    p1 = np.concatenate([[0, 0, 0], I])
    p2 = np.concatenate([[0.01, 0, 0], I])
    d1 = D.state_distance(p1, p2, "box", CUBE, pos_scale=0.01, rot_scale=0.1)
    d2 = D.state_distance(p1, p2, "box", CUBE, pos_scale=0.02, rot_scale=0.1)
    assert d1 == pytest.approx(1.0)
    assert d2 == pytest.approx(0.5)
