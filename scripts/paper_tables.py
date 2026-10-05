"""Markdown tables for the paper, straight from the committed result records.

No numbers are computed here: every cell is read from ``results/*/*.json``
(the files ``analyze.py`` reads), so a table can never disagree with the
decision table.  Writes ``results/PAPER_TABLES.md`` and prints it.

    python scripts/paper_tables.py [--results results] [--out results/PAPER_TABLES.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def _load(root: Path, *parts):
    p = root.joinpath(*parts)
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, prior=0.25):
    return f"{100 * x / prior:.1f} %"


def ci(s, scale=1.0, fmt="{:+.4f}"):
    return f"[{fmt.format(s['ci_low'] * scale)}, {fmt.format(s['ci_high'] * scale)}]"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=C.REPO_ROOT / "results")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    R = args.results
    sections = []

    e = _load(R, "exp_e", "results.json")
    d = _load(R, "exp_d", "floors.json")
    if e and d:
        prior = e["prior"]["pooled_mae_norm"]
        rows = []
        for v in ("A-std", "A-del", "A-time", "A-clip-del"):
            ent = e["variants"].get(v)
            if not ent:
                continue
            g = ent["G"]["pooled"]
            fl = d["variants"].get(v, {}).get("pooled", {}).get("floor_mae")
            rows.append([v, f"{fl:.5f}" if fl else "—",
                         f"{ent['conditions']['INTERACT']['point']:.5f} ({pct(ent['conditions']['INTERACT']['point'], prior)})",
                         f"{ent['conditions']['DECOY']['point']:.5f} ({pct(ent['conditions']['DECOY']['point'], prior)})",
                         f"{g['point']:+.5f} {ci(g)}",
                         f"{g['point'] / fl:+.3f} {ci(g, 1 / fl, '{:+.3f}')}" if fl else "—"])
        sections.append("## Experiment E — the confound (Architecture A, pooled geometries)\n\n"
                        "Prior (uniform-action) normalised MAE = 0.25.  G = mean of per-pair "
                        "[err(DECOY) − err(INTERACT)]; 95 % hierarchical-bootstrap CIs.\n\n"
                        + table(["IDM", "Floor (Exp D)", "err INTERACT", "err DECOY", "G", "G / floor"], rows))

    if d:
        rows = [[v, g, f"{s['floor_mae']:.5f}", pct(s['floor_mae'])]
                for v in sorted(d["variants"]) for g, s in d["variants"][v].items()]
        sections.append("## Experiment D — identifiability floors (INTERACT held-out)\n\n"
                        + table(["IDM", "Geometry", "Floor MAE", "% of prior"], rows))

    f = _load(R, "exp_f", "results.json")
    if f:
        mults = [f"{m:g}" for m in f["multipliers"]]
        rows = [[v] + [pct(f["variants"][v]["per_multiplier"][m]["point"]) for m in mults]
                + [f"{f['variants'][v]['spearman_rho']:+.2f}"] for v in sorted(f["variants"])]
        sections.append("## Experiment F — friction dose-response (% of prior)\n\n"
                        + table(["IDM"] + [f"×{m}" for m in mults] + ["Spearman"], rows))

    m = _load(R, "exp_masking", "results.json")
    if m and "cells" in m:
        rows = []
        for key in sorted(m["cells"]):
            v, mode = key.split("|")
            c = m["cells"][key]
            rows.append([v, {"full": "full frame", "object_masked": "arm pixels only",
                             "arm_masked": "object pixels only"}.get(mode, mode),
                         f"{c['point']:.5f}" if "point" in c else "—",
                         f"{100 * c['over_prior']:.1f} %"])
        sections.append("## Masking decomposition (Architecture B, descriptive)\n\n"
                        + table(["Probe", "Input", "MAE", "% of prior"], rows))

    e0 = _load(R, "exp_0", "results.json")
    if e0:
        rows = []
        for g in C.GEOMETRIES:
            d0 = e0.get(g, {})
            def f0(k):
                x = d0.get(k, {})
                return (f"{100 * x['mae_over_prior']:.1f} %"
                        if isinstance(x, dict) and "mae_over_prior" in x else "—")
            rows.append([g, f0("arm_only__INTERACT__s_std"), f0("object_only__INTERACT__s_del"),
                         f0("arm_only__INTERACT__s_del")])
        sections.append("## Experiment 0 — state oracles (Gate C0), % of prior\n\n"
                        + table(["Geometry", "arm-only @ s_std", "object-only @ s_del", "arm-only @ s_del"], rows))

    h = _load(R, "exp_h", "results.json")
    if h:
        lad = h.get("ladder_monotone", {})
        names = lad.get("models", list(h.get("within_model", {})))
        gts = dict(zip(names, lad.get("ground_truth_means", [])))
        ds_ = h.get("domain_shift", {}); dg = h.get("domain_gap", {}); wm = h.get("within_model", {})
        # Per-item scores (results/exp_h/ladder.npz, versioned): the OG-AF means.
        import numpy as np
        lp = R / "exp_h" / "ladder.npz"
        lz = np.load(lp, allow_pickle=False) if lp.exists() else None
        rows = []
        for n in names:
            s, g = ds_.get(n, {}), dg.get(n, {})
            og = (pct(float(lz[f"{n}_ogaf"].mean())) if lz is not None and f"{n}_ogaf" in lz.files else "—")
            rows.append([n, f"{gts[n]:.5f}" if n in gts else "—",
                         pct(s["standard_augmented"]) if s else "—",
                         og,
                         f"{s['dino_generated_vs_real']:.4f}" if s else "—",
                         f"{wm[n]['spearman']:+.3f}" if n in wm else "—",
                         pct(g["standard"]) if "standard" in g else "—",
                         pct(g["ogaf"]) if "ogaf" in g else "—"])
        sections.append("## Experiment H — world-model ladder (box, 193 held-out tuples per model)\n\n"
                        f"Ladder monotone under privileged ground-truth error: Spearman "
                        f"{lad.get('spearman', float('nan')):+.3f} vs 0.9 → "
                        f"{'PASS' if lad.get('passed') else 'FAIL; C3(b) between-model claims are void (§9-H)'}.  "
                        "C3(a) passes when the standard and OG-AF rankings correlate below 0.9.  "
                        "ABSENT = appearance-gap control (no object: physics divergence is zero by construction).\n\n"
                        + table(["Model", "GT error", "standard (gen)", "OG-AF (gen)", "DINO(gen,real)",
                                 "C3(a) ρ", "standard (ABSENT)", "OG-AF (ABSENT)"], rows))

    doc = "# Paper tables (generated by scripts/paper_tables.py — do not edit)\n\n" + "\n\n".join(sections) + "\n"
    out = args.out or (R / "PAPER_TABLES.md")
    out.write_text(doc)
    print(doc)
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
