"""Contact sheet: 3 geometries x 3 conditions x 4 horizons, for human inspection.

Build-order step 2 (§16) ends at a human checkpoint.  This renders the grid the
human looks at, plus a segmentation-overlay version so the §8.4 masks can be
eyeballed at the same time.

    python scripts/contact_sheet.py --out results/contact_sheet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import config as C  # noqa: E402
import scene  # noqa: E402

HORIZONS = ("s_0", "s_std", "s_del", "s_time")
LABEL_H = 14


def _label_strip(width: int, text: str) -> np.ndarray:
    """Tiny 5x7 bitmap font strip so the sheet is self-describing."""
    font = {
        "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        "B": ["11110", "10001", "11110", "10001", "10001", "10001", "11110"],
        "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
        "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
        "E": ["11111", "10000", "11110", "10000", "10000", "10000", "11111"],
        "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
        "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        "R": ["11110", "10001", "11110", "10100", "10010", "10001", "10001"],
        "S": ["01111", "10000", "01110", "00001", "00001", "10001", "01110"],
        "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
        "Y": ["10001", "01010", "00100", "00100", "00100", "00100", "00100"],
        "X": ["10001", "01010", "00100", "00100", "00100", "01010", "10001"],
        "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
        "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
        "M": ["10001", "11011", "10101", "10001", "10001", "10001", "10001"],
        "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
        "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
        "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
        " ": ["00000"] * 7,
    }
    strip = np.zeros((LABEL_H, width, 3), dtype=np.uint8)
    x = 3
    for ch in text.upper():
        glyph = font.get(ch, font[" "])
        for r, row in enumerate(glyph):
            for c, bit in enumerate(row):
                if bit == "1" and x + c < width:
                    strip[r + 3, x + c] = 255
        x += 6
    return strip


def _seg_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    out[mask == C.SEG_ARM] = 0.45 * out[mask == C.SEG_ARM] + 0.55 * np.array([255, 60, 60])
    out[mask == C.SEG_OBJECT] = 0.45 * out[mask == C.SEG_OBJECT] + 0.55 * np.array([60, 255, 120])
    return out.clip(0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "contact_sheet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-seed", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    action = scene.sample_action(np.random.default_rng(args.action_seed))
    print(f"action = {np.round(action, 4).tolist()}")

    S = C.RENDER_SIZE
    rows_rgb, rows_seg = [], []
    for geometry in C.GEOMETRIES:
        for condition in C.CONDITIONS:
            r = scene.rollout(action, seed=args.seed, condition=condition, geometry=geometry)
            fails = [k for k, v in r["validators"].items()
                     if isinstance(v, dict) and not v["passed"]]
            print(f"{geometry:9s} {condition:9s} contact_end={r['contact_end']:5d} "
                  f"s_del={r['horizon_idx']['s_del']:5d} fails={fails}")
            tiles_rgb = [_label_strip(S, f"{geometry[:3]} {condition[:3]}")]
            tiles_seg = [_label_strip(S, f"{geometry[:3]} {condition[:3]}")]
            row_r, row_s = [], []
            for h in HORIZONS:
                row_r.append(r["frames"][h])
                row_s.append(_seg_overlay(r["frames"][h], r["seg_masks"][h]))
            rows_rgb.append(np.concatenate(
                [np.concatenate([tiles_rgb[0]] + [_label_strip(S, h) for h in HORIZONS[1:]], 1)
                 if False else np.concatenate(
                     [_label_strip(S, f"{geometry[:3]}{condition[:3]} {h}") for h in HORIZONS], 1),
                 np.concatenate(row_r, 1)], 0))
            rows_seg.append(np.concatenate(
                [np.concatenate(
                    [_label_strip(S, f"{geometry[:3]}{condition[:3]} {h}") for h in HORIZONS], 1),
                 np.concatenate(row_s, 1)], 0))

    import imageio.v2 as imageio

    sheet = np.concatenate(rows_rgb, 0)
    seg_sheet = np.concatenate(rows_seg, 0)
    imageio.imwrite(args.out / "contact_sheet.png", sheet)
    imageio.imwrite(args.out / "contact_sheet_seg.png", seg_sheet)
    print(f"\nwrote {args.out / 'contact_sheet.png'}  {sheet.shape}")
    print(f"wrote {args.out / 'contact_sheet_seg.png'}")

    print("\n--- scene-level checks ---")
    for g in C.GEOMETRIES:
        print(scene.assert_corridor_decoy_disjoint(g))
    print(scene.check_rest_pose_occlusion("box", n_probe=48))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
