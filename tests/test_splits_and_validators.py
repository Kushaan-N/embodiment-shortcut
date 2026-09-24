"""Split integrity (T7) and validator logic on synthetic rollouts.

The split tests are the important half: T7 is a leak that would bias G *toward*
the hypothesis, so it must be impossible rather than merely unlikely.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C  # noqa: E402
import datasets  # noqa: E402


# ==========================================================================
# Splits (T7)
# ==========================================================================


def test_split_is_deterministic():
    a = [datasets.split_for_tuple("box", i) for i in range(500)]
    b = [datasets.split_for_tuple("box", i) for i in range(500)]
    assert a == b


def test_split_does_not_depend_on_condition():
    """Structural T7 guarantee: the API has no condition argument at all, so a
    condition-dependent split is not expressible."""
    import inspect

    params = set(inspect.signature(datasets.split_for_tuple).parameters)
    assert "condition" not in params
    assert params == {"geometry", "tuple_index", "salt"}


def test_all_condition_variants_of_a_tuple_share_a_split():
    for i in range(300):
        s = {datasets.split_for_tuple("box", i) for _ in C.CONDITIONS}
        assert len(s) == 1


def test_split_fractions_are_approximately_right():
    n = 20_000
    c = Counter(datasets.split_for_tuple("box", i) for i in range(n))
    train, val, test = C.SPLIT_FRACTIONS
    assert c["train"] / n == pytest.approx(train, abs=0.01)
    assert c["val"] / n == pytest.approx(val, abs=0.01)
    assert c["test"] / n == pytest.approx(test, abs=0.01)


def test_split_is_stable_under_corpus_extension():
    """Shards must be extendable without re-simulating or re-assigning."""
    first = [datasets.split_for_tuple("box", i) for i in range(1000)]
    # ... later the corpus grows to 5000 tuples ...
    later = [datasets.split_for_tuple("box", i) for i in range(5000)]
    assert later[:1000] == first


def test_split_differs_across_geometries():
    a = [datasets.split_for_tuple("box", i) for i in range(400)]
    b = [datasets.split_for_tuple("sphere", i) for i in range(400)]
    assert a != b


def test_split_salt_changes_the_assignment():
    a = [datasets.split_for_tuple("box", i, salt="x") for i in range(400)]
    b = [datasets.split_for_tuple("box", i, salt="y") for i in range(400)]
    assert a != b


def test_split_counts_helper():
    counts = datasets.split_counts("box", 2000)
    assert sum(counts.values()) == 2000
    assert all(v > 0 for v in counts.values())


# -------------------------------------------------- materialised-record check


def _records(assignments):
    return [{"geometry": g, "tuple_index": t, "split": s} for g, t, s in assignments]


def test_assert_no_tuple_spans_splits_passes_on_clean_data():
    recs = _records([("box", i, datasets.split_for_tuple("box", i))
                     for i in range(200) for _ in C.CONDITIONS])
    out = datasets.assert_no_tuple_spans_splits(recs)
    assert out["passed"]
    assert out["n_tuples_checked"] == 200


def test_assert_no_tuple_spans_splits_catches_a_leak():
    """The exact T7 failure: an INTERACT rollout in train, its twin in test."""
    recs = _records([("box", 5, "train"), ("box", 5, "test"), ("box", 6, "train")])
    out = datasets.assert_no_tuple_spans_splits(recs)
    assert not out["passed"]
    assert out["value"] == 1.0


def test_split_unit_is_declared_as_the_tuple():
    assert C.SPLIT_UNIT == "action_seed_tuple"


# ==========================================================================
# Validator logic on synthetic rollouts
# ==========================================================================


def test_compute_contact_end_finds_last_nonzero():
    import scene

    f = np.zeros(100)
    f[20:41] = 3.0
    assert scene.compute_contact_end(f, 1e-6) == 40


def test_compute_contact_end_reports_no_contact_as_negative():
    import scene

    assert scene.compute_contact_end(np.zeros(50), 1e-6) == -1


def test_compute_settled_at_requires_a_sustained_quiet_window():
    """A single quiet step can be the turning point of a bounce."""
    import scene

    vel = np.ones((400, 6)) * 1.0
    vel[100] = 0.0            # one isolated quiet step -- must NOT count
    vel[200:] = 0.0           # genuine settling
    got = scene.compute_settled_at(vel, start_idx=0, lin_thresh=0.1, ang_thresh=0.1,
                                   hold_steps=25)
    assert got == 200


def test_compute_settled_at_returns_none_when_never_settling():
    import scene

    vel = np.ones((400, 6))
    assert scene.compute_settled_at(vel, 0, 0.1, 0.1, hold_steps=25) is None


def test_compute_settled_at_ignores_quiet_before_start_idx():
    import scene

    vel = np.zeros((400, 6))
    vel[50:150] = 5.0
    got = scene.compute_settled_at(vel, start_idx=60, lin_thresh=0.1, ang_thresh=0.1,
                                   hold_steps=25)
    assert got == 150


# ==========================================================================
# Commanded trajectory invariants
# ==========================================================================


def test_commanded_path_is_independent_of_condition_by_construction():
    """T9: the trajectory function has no condition or object argument."""
    import inspect

    import scene

    params = set(inspect.signature(scene.commanded_path).parameters)
    assert "condition" not in params and "geometry" not in params


def test_commanded_path_starts_and_ends_at_canonical_rest():
    import scene

    p = scene.commanded_path([0.2, 0.3, 0.01], "retract")
    rest = np.array([*C.REST_POSE, C.REST_YAW])
    assert np.allclose(p[0], rest)
    assert np.allclose(p[-1], rest)


def test_eps_hold_arm_shares_the_prefix_through_the_withdrawal():
    import scene

    a = [0.2, 0.3, 0.01]
    pr = scene.commanded_path(a, "retract")
    pe = scene.commanded_path(a, "eps_hold")
    n = C.PHASES.push_end_idx + C.PHASES.eps_withdraw + 1
    assert np.allclose(pr[:n], pe[:n])
    assert not np.allclose(pr[-1], pe[-1])


def test_push_phase_is_constant_velocity():
    """The commanded speed IS the action; easing it would destroy the mapping."""
    import scene

    v = 0.3
    p = scene.commanded_path([v, 0.0, 0.0], "retract")
    ph = C.PHASES
    seg = p[ph.push_start_idx : ph.push_end_idx + 1, :2]
    steps = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    assert np.allclose(steps, v * C.SIM_DT, atol=1e-12)


def test_arm_state_at_s_std_is_injective_in_the_action():
    """C0 depends on this: (v, theta, d) -> (x, y, yaw) must be invertible."""
    import scene

    rng = np.random.default_rng(0)
    lo, hi = C.action_ranges_array()
    seen = []
    for _ in range(300):
        a = rng.uniform(lo, hi)
        p = scene.commanded_path(a, "retract")[C.PHASES.push_end_idx]
        seen.append((a, np.array([p[0], p[1], p[3]])))
    for i in range(0, 60):
        for j in range(i + 1, 60):
            da = np.abs((seen[i][0] - seen[j][0]) / (hi - lo)).max()
            ds = np.abs(seen[i][1] - seen[j][1]).max()
            if da > 0.05:
                assert ds > 1e-4, "distinct actions produced the same arm state at s_std"


def test_eps_withdrawal_is_along_minus_approach_direction():
    import scene

    theta = 0.7
    p = scene.commanded_path([0.2, theta, 0.0], "eps_hold")
    ph = C.PHASES
    u, _ = scene.approach_frame(theta)
    delta = p[ph.push_end_idx + ph.eps_withdraw, :2] - p[ph.push_end_idx, :2]
    assert np.dot(delta, u) < 0
    assert np.linalg.norm(delta) == pytest.approx(C.EPS_WITHDRAW, rel=1e-6)


def test_approach_frame_is_orthonormal():
    import scene

    for th in np.linspace(-math.pi / 2, math.pi / 2, 25):
        u, n = scene.approach_frame(th)
        assert np.linalg.norm(u) == pytest.approx(1.0)
        assert np.linalg.norm(n) == pytest.approx(1.0)
        assert abs(float(np.dot(u, n))) < 1e-12


# ==========================================================================
# Clip index sets (§8.1)
# ==========================================================================


def test_astd_clip_spans_the_action_execution_window():
    import scene

    idx = scene.clip_indices("A-std", s_del_idx=900)
    assert idx[0] == 0
    assert idx[-1] == C.PHASES.push_end_idx
    assert len(idx) <= C.CLIP_LENGTH
    assert (np.diff(idx) > 0).all()


def test_astd_clip_does_not_depend_on_s_del():
    import scene

    assert np.array_equal(scene.clip_indices("A-std", 700),
                          scene.clip_indices("A-std", 1200))


def test_clip_del_ends_at_the_actual_s_del_frame():
    import scene

    idx = scene.clip_indices("A-clip-del", s_del_idx=1000)
    assert idx[-1] == 1000


def test_clip_del_has_exactly_clip_length_frames_even_at_arm_rest():
    """s_del == arm_rest_idx for nearly every rollout (the object settles
    before the arm is home).  The old head ended AT arm_rest_idx, so np.unique
    merged it with s_del and the clip had 15 frames -- a collate error in the
    first mixed batch, and the T8 ablation could never have trained."""
    import scene

    ar = C.PHASES.arm_rest_idx
    for s_del in (ar, ar + 1, ar + 37, 1000, C.PHASES.max_steps - 1):
        idx = scene.clip_indices("A-clip-del", s_del_idx=s_del)
        assert len(idx) == C.CLIP_LENGTH, (s_del, idx)
        assert idx[0] == 0 and idx[-1] == s_del
        assert (np.diff(idx) > 0).all()


def test_clip_variants_sample_different_frames():
    import scene

    a = scene.clip_indices("A-std", s_del_idx=1000)
    b = scene.clip_indices("A-clip-del", s_del_idx=1000)
    assert not np.array_equal(a, b)
    assert C.PHASES.push_end_idx in a.tolist()
    assert 1000 not in a.tolist()


def test_pair_variants_are_rejected_as_clip_variants():
    import scene

    with pytest.raises(ValueError):
        scene.clip_indices("A-del", 800)


# ==========================================================================
# Action space
# ==========================================================================


def test_action_dims_are_non_circular():
    """§7.3's prior baseline is undefined for a wrapped variable."""
    lo, hi = C.ACTION_RANGES["approach_angle"]
    assert hi - lo < 2 * math.pi


def test_decoy_placement_is_outside_every_push_corridor():
    import scene

    for g in C.GEOMETRIES:
        out = scene.assert_corridor_decoy_disjoint(g)
        assert out["passed"], out


# ==========================================================================
# Masking decomposition (§8.4)
# ==========================================================================


def test_composite_mask_keeps_only_the_requested_label():
    import idm

    frames = np.full((4, 2, 8, 8, 3), 200, np.uint8)
    masks = np.zeros((4, 2, 8, 8), np.uint8)
    masks[..., 2:5, 2:5] = C.SEG_OBJECT
    bg = np.full((8, 8, 3), 30, np.uint8)
    out = idm.composite_mask(frames, masks, bg, C.SEG_OBJECT)
    assert out.shape == frames.shape
    assert (out[..., 2:5, 2:5, :] == 200).all()      # kept region survives
    assert (out[..., 0, 0, :] == 30).all()           # elsewhere is background


def test_composite_mask_accepts_mask_with_or_without_channel_axis():
    """Boolean fancy-indexing does not broadcast a trailing size-1 axis."""
    import idm

    frames = np.full((2, 4, 4, 3), 100, np.uint8)
    masks = np.zeros((2, 4, 4), np.uint8)
    masks[:, 1, 1] = C.SEG_ARM
    bg = np.zeros((4, 4, 3), np.uint8)
    a = idm.composite_mask(frames, masks, bg, C.SEG_ARM)
    b = idm.composite_mask(frames, masks[..., None], bg, C.SEG_ARM)
    assert np.array_equal(a, b)


def test_composite_mask_fills_with_background_not_black():
    """§8.4: a black fill lets the probe read mask SHAPE instead of content."""
    import idm

    frames = np.full((1, 4, 4, 3), 250, np.uint8)
    masks = np.zeros((1, 4, 4), np.uint8)
    bg = np.full((4, 4, 3), 77, np.uint8)
    out = idm.composite_mask(frames, masks, bg, C.SEG_OBJECT)
    assert (out == 77).all() and not (out == 0).any()


def test_masked_frames_modes_are_complementary():
    import idm_data as idd

    frames = np.full((1, 2, 6, 6, 3), 200, np.uint8)
    masks = np.zeros((1, 2, 6, 6), np.uint8)
    masks[..., 0:2, :] = C.SEG_ARM
    masks[..., 4:6, :] = C.SEG_OBJECT
    bg = np.zeros((6, 6, 3), np.uint8)
    arm_masked = idd.masked_frames(frames, masks, "arm_masked", bg)      # object only
    obj_masked = idd.masked_frames(frames, masks, "object_masked", bg)   # arm only
    assert (arm_masked[..., 4:6, :, :] == 200).all()
    assert (arm_masked[..., 0:2, :, :] == 0).all()
    assert (obj_masked[..., 0:2, :, :] == 200).all()
    assert (obj_masked[..., 4:6, :, :] == 0).all()
    assert np.array_equal(idd.masked_frames(frames, masks, "full", bg), frames)
