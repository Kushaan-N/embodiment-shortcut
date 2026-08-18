"""Content validation for cluster outputs (§13.5).

Runs inline per item (imported by the generator) and standalone over a finished
directory.

Governing principle: **the expensive failure is not a crash -- it is a job that
exits 0 and writes unusable output.  Validate content, never exit codes.**

    python unity/validate.py --dir data/generated/WM-base/box
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULTS = {
    "pixel_std_floor": 2.0,       # blank output
    "frame_diff_floor": 1.0,      # frozen output
    "frame0_max_delta": 40.0,     # wrong or dropped conditioning
}


def validate_array_item(path: Path, *, expected_frames: int = 1, **kw) -> dict:
    cfg = {**DEFAULTS, **kw}
    checks: dict = {}

    def add(name, passed, value, threshold):
        checks[name] = {"passed": bool(passed), "value": float(value),
                        "threshold": float(threshold)}

    try:
        with np.load(path) as z:
            gen = z["generated"]
            cond = z["conditioning"]
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "readable": False,
                "error": f"{type(exc).__name__}: {exc}", "_all_passed": False}

    frames = gen if gen.ndim == 4 else gen[None]
    add("frame_count_exact", len(frames) == expected_frames, len(frames), expected_frames)
    stds = [float(f.astype(np.float64).std()) for f in frames]
    add("pixel_std_above_floor", min(stds) > cfg["pixel_std_floor"],
        min(stds), cfg["pixel_std_floor"])
    if len(frames) > 1:
        diffs = [float(np.abs(frames[i + 1].astype(np.int32)
                              - frames[i].astype(np.int32)).mean())
                 for i in range(len(frames) - 1)]
        add("consecutive_frame_difference", min(diffs) > cfg["frame_diff_floor"],
            min(diffs), cfg["frame_diff_floor"])
    c0 = cond[0] if cond.ndim == 4 else cond
    d0 = float(np.abs(frames[0].astype(np.int32) - c0.astype(np.int32)).mean())
    add("frame0_close_to_conditioning", d0 < cfg["frame0_max_delta"],
        d0, cfg["frame0_max_delta"])

    digest = hashlib.sha256(np.ascontiguousarray(gen).tobytes()).hexdigest()
    return {"path": str(path), "readable": True, "sha256": digest, "checks": checks,
            "_all_passed": all(v["passed"] for v in checks.values())}


def validate_dir(d: Path, *, expected_frames: int = 1, expected_count: int | None = None,
                 **kw) -> dict:
    items = sorted(Path(d).glob("*.npz"))
    rows = [validate_array_item(p, expected_frames=expected_frames, **kw) for p in items]
    failed = [r for r in rows if not r["_all_passed"]]
    reasons: Counter = Counter()
    for r in failed:
        if not r.get("readable"):
            reasons["unreadable"] += 1
            continue
        for name, c in r["checks"].items():
            if not c["passed"]:
                reasons[name] += 1
    out = {
        "dir": str(d), "n_items": len(rows), "n_failed": len(failed),
        "failure_rate": len(failed) / max(len(rows), 1),
        "failure_reasons": dict(reasons),
        "expected_count": expected_count,
        "count_matches": None if expected_count is None else len(rows) == expected_count,
        "items": [{"path": r["path"], "sha256": r.get("sha256"),
                   "passed": r["_all_passed"]} for r in rows],
    }
    out["passed"] = bool(out["failure_rate"] <= 0.05
                         and (out["count_matches"] is not False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--expected-frames", type=int, default=1)
    ap.add_argument("--expected-count", type=int, default=None)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="write manifest.json here (content checksums, §13.5)")
    args = ap.parse_args()

    rep = validate_dir(args.dir, expected_frames=args.expected_frames,
                       expected_count=args.expected_count)
    print(f"{rep['n_items']} items, {rep['n_failed']} failed "
          f"({rep['failure_rate']:.1%})")
    for k, v in rep["failure_reasons"].items():
        print(f"  {k}: {v}")
    if rep["count_matches"] is False:
        print(f"  COUNT MISMATCH: expected {args.expected_count}, found {rep['n_items']} "
              f"-- silent truncation")
    out = args.manifest or (Path(args.dir) / "manifest.json")
    out.write_text(json.dumps(rep, indent=1))
    print(f"wrote {out}")
    if rep["failure_rate"] > 0.05:
        print("failure rate above 5% -- diagnose before analysing (§13.5)")
    return 0 if rep["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
