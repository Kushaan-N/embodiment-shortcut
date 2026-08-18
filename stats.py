"""Statistical machinery for §7.4 and §10 -- pre-registered, no interpretation.

Two things this module enforces structurally:

* **Paired contrasts.** ``paired_gap`` refuses to accept unmatched data.  The
  object-sensitivity gap G is the *mean of per-pair differences*; computing it
  as a difference of condition means throws away the design's power (§7.4, §14)
  and is not reachable through this API.
* **Hierarchical bootstrap.** Resample (action, seed) tuples within each
  training seed, then resample seeds -- 10,000 replicates (§10.1).  Seed-level
  min/max is returned alongside, because 5 seeds is too thin to bootstrap over
  alone and the honest presentation is both.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "PairedData", "BootstrapResult", "paired_gap", "hierarchical_bootstrap",
    "one_sided_p", "holm_correction", "cohen_style_effect", "summarise",
]


# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedData:
    """Per-pair values grouped by training seed.

    ``values[s]`` is a 1-D array of per-(action, seed) values produced by
    training seed ``s``.  Pair identity is carried in ``pair_ids[s]`` so the
    pairing can be re-verified downstream.
    """

    values: dict
    pair_ids: dict

    def __post_init__(self) -> None:
        if set(self.values) != set(self.pair_ids):
            raise ValueError("values and pair_ids must cover the same seeds")
        for s in self.values:
            if len(self.values[s]) != len(self.pair_ids[s]):
                raise ValueError(f"seed {s}: values/pair_ids length mismatch")
            if len(self.values[s]) == 0:
                raise ValueError(f"seed {s}: no pairs")

    @property
    def seeds(self) -> list:
        return sorted(self.values)

    def pooled(self) -> np.ndarray:
        return np.concatenate([np.asarray(self.values[s], float) for s in self.seeds])

    def seed_means(self) -> dict:
        return {s: float(np.mean(self.values[s])) for s in self.seeds}


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_boot: int
    seed_means: dict
    seed_min: float
    seed_max: float
    n_seeds: int
    n_pairs_total: int
    boot_std: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seed_means"] = {str(k): v for k, v in self.seed_means.items()}
        return d


# --------------------------------------------------------------------------


def paired_gap(
    err_by_condition: dict,
    pair_ids_by_condition: dict,
    *,
    minuend: str = "DECOY",
    subtrahend: str = "INTERACT",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair difference ``err(minuend) - err(subtrahend)`` (§7.4).

    Only pairs present in *both* conditions contribute.  Returns
    ``(differences, pair_ids)`` ordered by pair id.
    """
    for c in (minuend, subtrahend):
        if c not in err_by_condition:
            raise KeyError(f"missing condition {c!r}")
    a_ids = np.asarray(pair_ids_by_condition[minuend])
    b_ids = np.asarray(pair_ids_by_condition[subtrahend])
    a_val = np.asarray(err_by_condition[minuend], dtype=np.float64)
    b_val = np.asarray(err_by_condition[subtrahend], dtype=np.float64)
    if len(a_ids) != len(a_val) or len(b_ids) != len(b_val):
        raise ValueError("ids/values length mismatch")
    if len(set(a_ids.tolist())) != len(a_ids) or len(set(b_ids.tolist())) != len(b_ids):
        raise ValueError("duplicate pair ids within a condition")

    common = np.intersect1d(a_ids, b_ids)
    if common.size == 0:
        raise ValueError("no matched pairs between conditions -- pairing is broken")
    ia = {pid: i for i, pid in enumerate(a_ids.tolist())}
    ib = {pid: i for i, pid in enumerate(b_ids.tolist())}
    order = sorted(common.tolist())
    diffs = np.array([a_val[ia[p]] - b_val[ib[p]] for p in order], dtype=np.float64)
    return diffs, np.asarray(order)


def hierarchical_bootstrap(
    data: PairedData,
    statistic: Callable[[np.ndarray], float] | None = None,
    *,
    n_boot: int = 10_000,
    ci_level: float = 0.95,
    rng_seed: int = 987654321,
) -> BootstrapResult:
    """Resample seeds, then pairs within each resampled seed (§10.1).

    ``statistic`` maps the pooled resampled values to a scalar; default is the
    mean.  Percentile CIs, as pre-registered.
    """
    statistic = statistic or (lambda x: float(np.mean(x)))
    rng = np.random.default_rng(rng_seed)
    seeds = data.seeds
    n_seeds = len(seeds)
    arrays = [np.asarray(data.values[s], dtype=np.float64) for s in seeds]

    point = statistic(data.pooled())

    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.integers(0, n_seeds, size=n_seeds)
        parts = []
        for si in chosen:
            arr = arrays[si]
            idx = rng.integers(0, arr.size, size=arr.size)
            parts.append(arr[idx])
        boot[b] = statistic(np.concatenate(parts))

    alpha = 1.0 - ci_level
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    sm = data.seed_means()
    return BootstrapResult(
        point=float(point),
        ci_low=float(lo),
        ci_high=float(hi),
        ci_level=ci_level,
        n_boot=n_boot,
        seed_means=sm,
        seed_min=float(min(sm.values())),
        seed_max=float(max(sm.values())),
        n_seeds=n_seeds,
        n_pairs_total=int(sum(a.size for a in arrays)),
        boot_std=float(np.std(boot, ddof=1)),
    )


def one_sided_p(
    data: PairedData,
    null_value: float,
    direction: str,
    *,
    n_boot: int = 10_000,
    rng_seed: int = 987654321,
) -> float:
    """Bootstrap one-sided p-value against ``null_value``.

    ``direction='less'`` tests H1: statistic < null_value.
    ``direction='greater'`` tests H1: statistic > null_value.
    """
    if direction not in ("less", "greater"):
        raise ValueError("direction must be 'less' or 'greater'")
    rng = np.random.default_rng(rng_seed)
    seeds = data.seeds
    arrays = [np.asarray(data.values[s], dtype=np.float64) for s in seeds]
    n_seeds = len(seeds)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, n_seeds, size=n_seeds)
        parts = []
        for si in chosen:
            arr = arrays[si]
            parts.append(arr[rng.integers(0, arr.size, size=arr.size)])
        boot[b] = float(np.mean(np.concatenate(parts)))
    if direction == "less":
        frac = float(np.mean(boot >= null_value))
    else:
        frac = float(np.mean(boot <= null_value))
    # +1 smoothing so p is never exactly 0 with a finite bootstrap.
    return (frac * n_boot + 1.0) / (n_boot + 1.0)


def holm_correction(pvalues: Sequence[float], labels: Sequence[str] | None = None) -> dict:
    """Holm-Bonferroni step-down (§10.2).  Returns adjusted p-values."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    labels = list(labels) if labels is not None else [f"test_{i}" for i in range(m)]
    return {lab: {"p_raw": float(p[i]), "p_holm": float(adj[i])} for i, lab in enumerate(labels)}


def cohen_style_effect(point: float, floor: float) -> float:
    """Effect size standardised by the matching Experiment D floor (§10.1)."""
    if floor <= 0:
        raise ValueError("floor must be positive")
    return float(point / floor)


def summarise(
    data: PairedData,
    *,
    floor: float | None = None,
    n_boot: int = 10_000,
    ci_level: float = 0.95,
    rng_seed: int = 987654321,
) -> dict:
    """Bootstrap CI + seed spread + optional floor-standardised effect size."""
    res = hierarchical_bootstrap(
        data, n_boot=n_boot, ci_level=ci_level, rng_seed=rng_seed
    )
    out = res.to_dict()
    if floor is not None and floor > 0:
        out["floor"] = float(floor)
        out["effect_over_floor"] = cohen_style_effect(res.point, floor)
        out["ci_low_over_floor"] = float(res.ci_low / floor)
        out["ci_high_over_floor"] = float(res.ci_high / floor)
    return out
