"""Unit tests for the §7.3 prior baselines and the §10 statistical plan.

The prior baseline is what makes every error number in the paper
interpretable, and the paired estimator is where the design's statistical power
comes from.  Both are analytic enough to test exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import distances as D  # noqa: E402
import stats  # noqa: E402


# ------------------------------------------------------------ prior baseline


@pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (-2.0, 5.0), (0.12, 0.36)])
def test_uniform_prior_mae_is_range_over_four(lo, hi):
    assert D.uniform_prior_mae(lo, hi) == pytest.approx((hi - lo) / 4)


@pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (-2.0, 5.0), (0.12, 0.36)])
def test_uniform_prior_mse_is_variance(lo, hi):
    assert D.uniform_prior_mse(lo, hi) == pytest.approx((hi - lo) ** 2 / 12)


def test_normalised_prior_is_exactly_a_quarter_for_any_uniform():
    for lo, hi in [(0.0, 1.0), (-3.0, 7.5), (100.0, 100.001)]:
        assert D.normalised_uniform_prior_mae(lo, hi) == pytest.approx(0.25)
        assert D.normalised_uniform_prior_mse(lo, hi) == pytest.approx(1 / 12)


def test_analytic_prior_matches_monte_carlo():
    """§7.3 requires the analytic and empirical priors to agree."""
    rng = np.random.default_rng(0)
    lo, hi = -1.5, 4.0
    x = rng.uniform(lo, hi, size=400_000)
    mid = 0.5 * (lo + hi)
    assert np.abs(x - mid).mean() == pytest.approx(D.uniform_prior_mae(lo, hi), rel=0.01)
    assert ((x - mid) ** 2).mean() == pytest.approx(D.uniform_prior_mse(lo, hi), rel=0.02)


def test_midpoint_is_the_optimal_constant_predictor_for_mae():
    rng = np.random.default_rng(1)
    lo, hi = 0.0, 1.0
    x = rng.uniform(lo, hi, size=200_000)
    best = D.uniform_prior_mae(lo, hi)
    for c in (0.2, 0.35, 0.5, 0.65, 0.8):
        mae = np.abs(x - c).mean()
        assert mae >= best - 0.003


def test_prior_baseline_table_shape():
    t = D.prior_baseline_table({"a": (0.0, 1.0), "b": (-1.0, 1.0)})
    assert t["pooled_mae_norm"] == pytest.approx(0.25)
    assert set(t["per_dim"]) == {"a", "b"}


# ------------------------------------------------------------- paired gap


def test_paired_gap_is_mean_of_differences_not_difference_of_means():
    """The two coincide only for balanced, fully-matched data.

    §14 lists 'computing G as a difference of means' as a known failure mode,
    so this test uses data where the two genuinely disagree.
    """
    err = {"DECOY": [10.0, 20.0, 30.0], "INTERACT": [1.0, 2.0]}
    ids = {"DECOY": [1, 2, 3], "INTERACT": [1, 2]}
    diffs, pair_ids = stats.paired_gap(err, ids)
    assert list(pair_ids) == [1, 2]
    assert diffs.tolist() == [9.0, 18.0]
    assert diffs.mean() == pytest.approx(13.5)
    naive = np.mean(err["DECOY"]) - np.mean(err["INTERACT"])
    assert naive == pytest.approx(20.0 - 1.5)
    assert abs(naive - diffs.mean()) > 1.0


def test_paired_gap_uses_only_matched_pairs():
    err = {"DECOY": [5.0, 6.0], "INTERACT": [1.0, 2.0]}
    ids = {"DECOY": [7, 8], "INTERACT": [8, 9]}
    diffs, pair_ids = stats.paired_gap(err, ids)
    assert list(pair_ids) == [8]
    assert diffs.tolist() == [6.0 - 1.0]


def test_paired_gap_rejects_unmatched_data():
    with pytest.raises(ValueError):
        stats.paired_gap({"DECOY": [1.0], "INTERACT": [1.0]},
                         {"DECOY": [1], "INTERACT": [2]})


def test_paired_gap_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        stats.paired_gap({"DECOY": [1.0, 2.0], "INTERACT": [1.0, 2.0]},
                         {"DECOY": [1, 1], "INTERACT": [1, 2]})


# ---------------------------------------------------- hierarchical bootstrap


def _synthetic(seed_effect, n_seeds=5, n_pairs=200, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    values, ids = {}, {}
    for s in range(n_seeds):
        offset = seed_effect * (s - (n_seeds - 1) / 2)
        values[s] = rng.normal(1.0 + offset, 0.5, size=n_pairs)
        ids[s] = np.arange(n_pairs)
    return stats.PairedData(values=values, pair_ids=ids)


def test_bootstrap_point_estimate_is_the_pooled_mean():
    d = _synthetic(0.0)
    r = stats.hierarchical_bootstrap(d, n_boot=200)
    assert r.point == pytest.approx(d.pooled().mean())


def test_bootstrap_ci_covers_the_truth():
    d = _synthetic(0.0, rng_seed=3)
    r = stats.hierarchical_bootstrap(d, n_boot=2000, rng_seed=11)
    assert r.ci_low < 1.0 < r.ci_high


def test_seed_variation_widens_the_interval():
    """The hierarchical scheme must notice between-seed spread; a flat
    bootstrap over pooled samples would not."""
    narrow = stats.hierarchical_bootstrap(_synthetic(0.0, rng_seed=5),
                                          n_boot=2000, rng_seed=7)
    wide = stats.hierarchical_bootstrap(_synthetic(0.6, rng_seed=5),
                                        n_boot=2000, rng_seed=7)
    assert (wide.ci_high - wide.ci_low) > 2 * (narrow.ci_high - narrow.ci_low)


def test_seed_min_max_reported_alongside():
    r = stats.hierarchical_bootstrap(_synthetic(0.4), n_boot=200)
    assert r.n_seeds == 5
    assert r.seed_min < r.point < r.seed_max
    assert len(r.seed_means) == 5


def test_bootstrap_is_deterministic_given_the_seed():
    d = _synthetic(0.2)
    a = stats.hierarchical_bootstrap(d, n_boot=500, rng_seed=42)
    b = stats.hierarchical_bootstrap(d, n_boot=500, rng_seed=42)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_paired_data_rejects_ragged_input():
    with pytest.raises(ValueError):
        stats.PairedData(values={0: [1.0, 2.0]}, pair_ids={0: [1]})
    with pytest.raises(ValueError):
        stats.PairedData(values={0: []}, pair_ids={0: []})


# --------------------------------------------------------------- p-values


def test_one_sided_p_is_small_when_effect_is_clear():
    d = _synthetic(0.0)                       # mean ~ 1.0
    p = stats.one_sided_p(d, null_value=2.0, direction="less", n_boot=1000)
    assert p < 0.01


def test_one_sided_p_is_calibrated_under_the_null():
    """Under H0 the p-value should be roughly uniform.

    Asserting a band on a *single* draw would be testing the draw, not the
    estimator: with 5 seeds the bootstrap SE is seed-dominated and any single
    p in [0, 1] is legitimate.  Calibration over many independent datasets is
    the property that actually matters for the pre-registered tests.
    """
    ps = [
        stats.one_sided_p(_synthetic(0.0, rng_seed=k), null_value=1.0,
                          direction="less", n_boot=400, rng_seed=1000 + k)
        for k in range(40)
    ]
    ps = np.asarray(ps)
    assert 0.25 < ps.mean() < 0.75
    # A well-calibrated one-sided test rejects at alpha about alpha of the time.
    assert ps[ps < 0.05].size <= 6


def test_one_sided_p_is_never_exactly_zero():
    d = _synthetic(0.0)
    assert stats.one_sided_p(d, null_value=100.0, direction="less", n_boot=500) > 0


# ------------------------------------------------------------------ Holm


def test_holm_matches_hand_computation():
    out = stats.holm_correction([0.01, 0.04, 0.03], ["a", "b", "c"])
    assert out["a"]["p_holm"] == pytest.approx(0.03)   # 3 * 0.01
    assert out["c"]["p_holm"] == pytest.approx(0.06)   # 2 * 0.03
    assert out["b"]["p_holm"] == pytest.approx(0.06)   # max(0.06, 1 * 0.04)


def test_holm_is_monotone_and_bounded():
    ps = [0.001, 0.02, 0.2, 0.9]
    out = stats.holm_correction(ps)
    adj = [out[f"test_{i}"]["p_holm"] for i in range(4)]
    assert all(a <= b + 1e-12 for a, b in zip(adj, adj[1:]))
    assert all(0 <= a <= 1 for a in adj)


def test_holm_single_test_is_unchanged():
    out = stats.holm_correction([0.03], ["only"])
    assert out["only"]["p_holm"] == pytest.approx(0.03)


def test_effect_size_requires_positive_floor():
    with pytest.raises(ValueError):
        stats.cohen_style_effect(1.0, 0.0)


# ==========================================================================
# Pair ids must be unique once geometries are pooled


def test_pair_ids_are_unique_across_pooled_geometries_and_decode():
    import stats

    g = np.array(["box", "sphere", "cylinder", "box"])
    t = np.array([7, 7, 7, 8])
    ids = stats.pair_id(g, t)
    assert ids.dtype == np.int64
    assert len(set(ids.tolist())) == 4          # box/7 != sphere/7 != cylinder/7
    assert (stats.pair_id_geometry(ids) == g).all()
    assert (ids % stats._PAIR_ID_STRIDE == t).all()


def test_paired_gap_accepts_pooled_geometries_via_pair_id():
    """The pooled INTERACT/DECOY arrays of Experiment E repeat tuple_index
    across geometries; bare indices made paired_gap raise 'duplicate pair
    ids'.  Composite ids pair box/7 with box/7 only."""
    import stats

    g = np.array(["box", "sphere", "box", "sphere"])
    t = np.array([7, 7, 8, 8])
    ids = stats.pair_id(g, t)
    with pytest.raises(ValueError):
        stats.paired_gap({"DECOY": np.ones(4), "INTERACT": np.zeros(4)},
                         {"DECOY": t, "INTERACT": t})
    diffs, out_ids = stats.paired_gap({"DECOY": np.array([1., 2., 3., 4.]),
                                       "INTERACT": np.array([0., 0., 0., 0.])},
                                      {"DECOY": ids, "INTERACT": ids})
    assert len(diffs) == 4
    assert sorted(out_ids.tolist()) == sorted(ids.tolist())
