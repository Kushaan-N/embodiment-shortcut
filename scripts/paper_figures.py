"""Generate the paper's pgfplots figures from the committed result records.

Every coordinate is read from results/*/ (the files analyze.py reads), so a
figure cannot disagree with the decision table.  Writes paper/figures/*.tex.

    python scripts/paper_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper" / "figures"
PRIOR = 0.25


def L(*p):
    return json.loads(R.joinpath(*p).read_text())


def coords(pairs, err=None):
    if err is None:
        return " ".join(f"({x:.6g},{y:.6g})" for x, y in pairs)
    return " ".join(f"({x:.6g},{y:.6g}) +- (0,{e:.6g})" for (x, y), e in zip(pairs, err))


def friction():
    f = L("exp_f", "results.json")
    ks = f["multipliers"]
    s = []
    for v, style, lab in (("A-std", "blue, mark=square*", "Standard (A-std, $s_\\mathrm{std}$)"),
                          ("A-del", "red, mark=*", "OG-AF (A-del, $s_\\mathrm{del}$)")):
        pm = f["variants"][v]["per_multiplier"]
        pts = [(k, 100 * pm[f"{k:g}"]["point"] / PRIOR) for k in ks]
        err = [100 * (pm[f"{k:g}"]["ci_high"] - pm[f"{k:g}"]["ci_low"]) / 2 / PRIOR for k in ks]
        s.append(f"\\addplot+[{style}, error bars/.cd, y dir=both, y explicit] coordinates {{{coords(pts, err)}}};\n"
                 f"\\addlegendentry{{{lab}}}")
    return ("\\begin{tikzpicture}\\begin{axis}[width=\\linewidth, height=4.6cm, xmode=log, log basis x=2,\n"
            "  xlabel={object friction multiplier (trained at $\\times1$)}, ylabel={IDM error (\\% of prior)},\n"
            "  xtick={0.25,0.5,1,2,4}, xticklabels={0.25,0.5,1,2,4}, ymin=0, ymax=70, grid=major,\n"
            "  legend style={font=\\scriptsize, at={(0.98,0.10)}, anchor=south east}, tick label style={font=\\scriptsize},\n"
            "  label style={font=\\small}]\n"
            "\\draw[gray, dashed] (axis cs:1,0) -- (axis cs:1,70) node[pos=0.97, right, font=\\tiny] {trained};\n"
            + "\n".join(s) + "\n\\end{axis}\\end{tikzpicture}\n")


def resolution():
    r = L("exp_resolution", "results.json")
    s = []
    for v, style, lab in (("A-std", "blue, mark=square*", "Standard"), ("A-del", "red, mark=*", "OG-AF")):
        bs = [b for b in r["bins"] if v in b]
        pts = [(b["mean_disp_mm"], 100 * b[v]["mean_delta"] / PRIOR) for b in bs]
        err = [100 * (b[v]["ci"][1] - b[v]["ci"][0]) / 2 / PRIOR for b in bs]
        s.append(f"\\addplot+[{style}, error bars/.cd, y dir=both, y explicit] coordinates {{{coords(pts, err)}}};\n"
                 f"\\addlegendentry{{{lab}}}")
    return ("\\begin{tikzpicture}\\begin{axis}[width=\\linewidth, height=4.6cm, xmode=log,\n"
            "  xlabel={shift of settled object position vs.\\ $\\times1$ friction (mm)},\n"
            "  ylabel={$\\Delta$ IDM error (\\% of prior)}, grid=major, ymin=-5, ymax=65,\n"
            "  legend style={font=\\scriptsize, at={(0.02,0.98)}, anchor=north west}, tick label style={font=\\scriptsize},\n"
            "  label style={font=\\small}]\n"
            "\\draw[gray, dashed] (axis cs:4.9,-5) -- (axis cs:4.9,65) node[pos=0.55, right, font=\\tiny] {$\\delta_{\\mathrm{pos}}$};\n"
            + "\n".join(s) + "\n\\end{axis}\\end{tikzpicture}\n")


def h2():
    ci = h2_cis()
    sym = {"standard": "Standard", "ogaf": "OG-AF"}
    inter = " ".join(f"({sym[v]},{ci[(v, '')][0]:.3f}) +- (0,{ci[(v, '')][1]:.3f})" for v in sym)
    absent = " ".join(f"({sym[v]},{ci[(v, '_ABSENT')][0]:.3f}) +- (0,{ci[(v, '_ABSENT')][1]:.3f})"
                      for v in sym)
    return ("\\begin{tikzpicture}\\begin{axis}[width=\\linewidth, height=4.6cm, ybar, bar width=14pt,\n"
            "  symbolic x coords={Standard,OG-AF}, xtick=data, enlarge x limits=0.5, ymin=-5, ymax=58,\n"
            "  ylabel={$\\Delta$ error (\\% of prior)}, grid=major,\n"
            "  legend columns=1,\n"
            "  legend style={font=\\scriptsize, at={(0.5,1.03)}, anchor=south},\n"
            "  tick label style={font=\\scriptsize}, label style={font=\\small}]\n"
            f"\\addplot+[fill=black!60, draw=black, error bars/.cd, y dir=both, y explicit] coordinates {{{inter}}};\\addlegendentry{{INTERACT (physics matters)}}\n"
            f"\\addplot+[fill=black!15, draw=black, error bars/.cd, y dir=both, y explicit] coordinates {{{absent}}};\\addlegendentry{{ABSENT (no object: no physics)}}\n"
            "\\end{axis}\\end{tikzpicture}\n")


def conditions_png():
    """Method figure: the box rows (INTERACT, DECOY, ABSENT) of the committed
    contact sheet -- real renders at s_0, s_std, s_del, s_time."""
    import imageio.v2 as imageio

    src = R / "contact_sheet" / "contact_sheet.png"
    img = imageio.imread(src)
    row = img.shape[0] // 9                     # 3 geometries x 3 conditions
    bar = 14                                    # renderer's text label strip, illegible in print
    rows = [img[i * row + bar:(i + 1) * row] for i in range(3)]
    gap = np.full((4, img.shape[1], img.shape[2]), 255, dtype=img.dtype)
    imageio.imwrite(OUT / "conditions.png", np.concatenate([rows[0], gap, rows[1], gap, rows[2]]))
    print("wrote paper/figures/conditions.png")


def h2_cis():
    """Paired bootstrap CIs (tuples) for the Fig. 3 bars, same seed as the paper."""
    z = np.load(R / "exp_h2" / "ladder.npz", allow_pickle=False)
    b, c = "WM-base-100", "WM-physics-corrupted-v2"
    rng = np.random.default_rng(987654321)
    out = {}
    for v in ("standard", "ogaf"):
        for cond in ("", "_ABSENT"):
            A = dict(zip(z[f"{b}{cond}_tuple_index"].tolist(), z[f"{b}{cond}_{v}"]))
            B = dict(zip(z[f"{c}{cond}_tuple_index"].tolist(), z[f"{c}{cond}_{v}"]))
            d = np.array([B[t] - A[t] for t in sorted(set(A) & set(B))])
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
            out[(v, cond)] = (100 * d.mean() / PRIOR,
                              100 * (np.percentile(bs, 97.5) - np.percentile(bs, 2.5)) / 2 / PRIOR)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    conditions_png()
    for name, fn in (("friction", friction), ("resolution", resolution), ("h2", h2)):
        (OUT / f"{name}.tex").write_text("% GENERATED by scripts/paper_figures.py from results/ -- do not edit\n" + fn())
        print(f"wrote paper/figures/{name}.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
