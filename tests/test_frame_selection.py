"""The memmap horizon-selection optimisation must not change any pixel.

`train_idm.embed_stores` reads two horizons out of a four-horizon store with
`np.ix_` rather than `frames[rows][:, pos]`, which materialises all four and
discards half.  That is a pure I/O change and these tests pin it as one: if a
future edit reintroduces the slow form, or gets the axis order wrong, the
embeddings feeding every Architecture B probe would silently change.
"""

from __future__ import annotations

import numpy as np
import pytest

import idm_data as idd


def _fake_store(n=7, horizons=4, size=5, seed=0):
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, (n, horizons, size, size, 3), dtype=np.uint8)
    masks = rng.integers(0, 3, (n, horizons, size, size), dtype=np.uint8)
    return frames, masks


@pytest.mark.parametrize("pos", [[0, 1], [0, 2], [0, 3], [1, 3]])
def test_ix_selection_matches_naive_form(pos):
    frames, _ = _fake_store()
    rows = [0, 2, 3, 6]
    naive = np.asarray(frames[rows][:, pos])
    fast = np.asarray(frames[np.ix_(rows, pos)])
    assert np.array_equal(naive, fast)
    assert fast.shape == (len(rows), len(pos)) + frames.shape[2:]


def test_ix_selection_preserves_row_and_horizon_order():
    """Order matters: a transposed selection would still compare equal by sum."""
    frames, _ = _fake_store()
    rows, pos = [5, 1, 4], [2, 0]
    fast = np.asarray(frames[np.ix_(rows, pos)])
    for i, r in enumerate(rows):
        for j, p in enumerate(pos):
            assert np.array_equal(fast[i, j], frames[r, p])


@pytest.mark.parametrize("mode", list(idd.MASK_MODES))
def test_masking_agrees_under_both_selection_forms(mode):
    frames, masks = _fake_store()
    rows, pos = [1, 2, 5], [0, 2]
    bg = np.zeros(3, dtype=np.uint8)
    sel = np.ix_(rows, pos)

    a_f, a_m = np.asarray(frames[rows][:, pos]), np.asarray(masks[rows][:, pos])
    b_f, b_m = np.asarray(frames[sel]), np.asarray(masks[sel])
    if mode != "full":
        a_f = idd.masked_frames(a_f, a_m, mode, bg)
        b_f = idd.masked_frames(b_f, b_m, mode, bg)
    assert np.array_equal(a_f, b_f)


def test_selection_reads_fewer_bytes_than_naive_form(tmp_path):
    """The point of the change: touch two horizons, not four."""
    frames, _ = _fake_store(n=64, horizons=4, size=32)
    path = tmp_path / "frames.npy"
    np.save(path, frames)

    mm = np.load(path, mmap_mode="r")
    rows, pos = list(range(64)), [0, 2]
    naive = np.asarray(mm[rows][:, pos])
    fast = np.asarray(mm[np.ix_(rows, pos)])
    assert np.array_equal(naive, fast)
    # The intermediate the naive form builds is twice the size of the result.
    assert np.asarray(mm[rows]).nbytes == 2 * fast.nbytes


def test_store_paths_are_distinct_for_fractional_friction_multipliers():
    """Path.with_suffix ate the '.25' of '_fm0.25', so x0.25 and x0.5 shared one
    frame store and Experiment F scored both on the same data."""
    from pathlib import Path

    import idm_data as idd

    seen = {}
    for fm in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0):
        paths = idd._store_paths(Path("/tmp/fs"), "box", "INTERACT", fm)
        assert len({p.name for p in paths}) == 3
        for p in paths:
            assert p.name not in seen, (fm, p.name, seen.get(p.name))
            seen[p.name] = fm
    assert idd._store_paths(Path("/tmp/fs"), "box", "INTERACT", 0.25)[0].name \
        == "box_INTERACT_fm0.25.frames.npy"
