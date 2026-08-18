"""The only place that turns saved arrays into verdicts (§0.3, §12).

Reads ``.npz`` / ``results.json`` outputs.  For each gate it prints the measured
value, the threshold, PASS/FAIL, and the CI.  No interpretation, no hedging, no
plotting, and no recomputation of anything a compute script already saved.

``--exp e`` and ``--exp h`` are sealed behind the mechanical pre-registration
lock of §0.5.  The check lives in ``prereg_lock.require`` and has no skip flag.

    python analyze.py --exp all
    python analyze.py --exp e
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config as C
import distances as D
import prereg_lock
import stats

W = 78


def rule(title: str = "") -> None:
    print("\n" + "=" * W)
    if title:
        print(title)
        print("=" * W)


def verdict(name: str, measured, threshold, passed: bool, *, ci=None, units: str = "") -> None:
    ci_s = "" if ci is None else f"  95% CI [{ci[0]:.5f}, {ci[1]:.5f}]"
    m = f"{measured:.5f}" if isinstance(measured, (int, float, np.floating)) else str(measured)
    t = f"{threshold:.5f}" if isinstance(threshold, (int, float, np.floating)) else str(threshold)
    print(f"  {name:52s} {m:>12s}{units}  vs {t:>10s}{units}  "
          f"{'PASS' if passed else 'FAIL'}{ci_s}")


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# ==========================================================================


def analyze_0(root: Path) -> None:
    res = _load(root / "exp_0" / "results.json")
    rule("EXPERIMENT 0 -- state-space oracles (Gate C0, §9-0)")
    if res is None:
        print("  not run")
        return
    g = res["_gate_C0"]
    th = g["thresholds"]
    for geom, v in g["per_geometry"].items():
        print(f"  [{geom}]")
        verdict("C0 arm-only @ s_std (frac of prior)",
                v["arm_only_s_std_over_prior"], th["near_floor_frac_of_prior"],
                v["C0_shortcut_channel_exists"])
        verdict("object-only @ s_del (frac of prior)",
                v["object_only_s_del_over_prior"], th["at_prior_frac_of_prior"],
                v["object_recoverable_at_s_del"])
        verdict("arm channel INTERACT == DECOY @ s_std",
                abs(v["arm_only_s_std_over_prior"] - v["arm_only_DECOY_s_std_over_prior"]),
                0.05, v["arm_channel_condition_invariant"])
        verdict("T9 arm residual carries no action",
                v["arm_residual_s_std_over_prior"], th["at_prior_frac_of_prior"],
                v["T9_residual_carries_no_action"])
    print(f"\n  GATE C0: {'PASS' if g['passed'] else 'FAIL'}")


def analyze_a(root: Path) -> None:
    rule("EXPERIMENT A -- contact-regime deltas (§9-A)")
    thr = _load(root / "exp_a" / "thresholds.json")
    if thr is None:
        print("  not run")
        return
    print(f"  measured at stride {thr['frame_stride']} steps ({thr['video_fps']:g} fps), "
          f"{thr['percentile']:g}th percentile of nonzero active-phase deltas")
    for g in C.GEOMETRIES:
        d = thr["per_geometry_detail"][g]
        rot = ("n/a (rotation-invariant)" if d["delta_rot_min"] is None
               else f"{np.degrees(d['delta_rot_min']):.4f} deg")
        print(f"  {g:9s} delta_pos_min={d['delta_pos_min'] * 1e3:8.4f} mm   "
              f"delta_rot_min={rot:24s} settle_lin={d['settle_lin_speed'] * 1e3:7.2f} mm/s")
    print("  (these are the sole source of every downstream threshold; nothing is hardcoded)")


def analyze_b(root: Path) -> None:
    res = _load(root / "exp_b" / "results.json")
    rule("EXPERIMENT B -- encoder resolution floor (Gate B, §9-B, T4)")
    if res is None:
        print("  not run -- this is the FIRST GPU GATE; run it before anything else on a GPU")
        return
    for enc, pools in res["encoders"].items():
        for pool, v in pools.items():
            verdict(f"{enc.split('/')[-1]} / {pool}: translation @ delta_min",
                    v["translation_at_delta_min_sigma"], res["gate_sigma"],
                    v["passed_translation"], units=" sigma")
            verdict(f"{enc.split('/')[-1]} / {pool}: rotation @ delta_min",
                    v["rotation_at_delta_min_sigma"], res["gate_sigma"],
                    v["passed_rotation"], units=" sigma")
    for u in res.get("unavailable_encoders", []):
        print(f"  UNAVAILABLE (reported, not substituted): {u['encoder']}: {u['error']}")
    print(f"\n  GATE B: {'PASS' if res['_gate_B']['passed'] else 'FAIL'}")


def analyze_c(root: Path) -> None:
    res = _load(root / "exp_c" / "results.json")
    rule("EXPERIMENT C -- injectivity (Gate C, §9-C, T3)")
    if res is None:
        print("  not run")
        return
    eps = res["eps_a_working"]
    for g, v in res["_gate_C"]["per_geometry"].items():
        verdict(f"{g}: unidentifiable frac @ s_del (eps_a={eps:g})",
                v["s_del_quotiented"], res["gate_threshold"], v["passed"])
        print(f"    {'':52s} unquotiented {v['s_del_unquotiented']:.5f}, "
              f"s_std {v['s_std_quotiented']:.5f}; binding: {v['binding_distance']}")
    print(f"\n  GATE C: {'PASS' if res['_gate_C']['passed'] else 'FAIL'}")


def analyze_corpus(root: Path) -> None:
    res = _load(root / "validate_corpus" / "report.json")
    rule("CORPUS VALIDATION -- §6 validators, T11 gate, T7 split integrity")
    if res is None:
        print("  not run")
        return
    t11 = res["t11_gate"]
    verdict("T11 DECOY placement -> action, max ridge R^2",
            t11["value"], t11["threshold"], t11["passed"])
    sp = res["split_integrity"]
    verdict("T7 tuples spanning splits", sp["value"], 0.0, sp["passed"])
    for cell, v in sorted(res["validators"].items()):
        print(f"  {cell:22s} n={v['n_rollouts']:6d}  excluded={v['exclusion_rate'] * 100:5.2f}%")
        for name, c in sorted(v["checks"].items()):
            if c["n_failed"]:
                print(f"      {name:34s} {c['n_failed']:5d} failed "
                      f"({c['fail_rate'] * 100:.2f}%)  max={c['value_max']:.4g}")
    print(f"\n  CORPUS GATE: {'PASS' if res['passed'] else 'FAIL'}")


def analyze_d(root: Path) -> None:
    res = _load(root / "exp_d" / "floors.json")
    rule("EXPERIMENT D -- identifiability floors, INTERACT held-out only (§9-D)")
    if res is None:
        print("  not run")
        return
    prior = res["prior"]["pooled_mae_norm"]
    print(f"  prior baseline (normalised MAE) = {prior:.5f}\n")
    print(f"  {'variant':14s}{'geometry':10s}{'floor':>10s}{'% of prior':>12s}"
          f"{'seed min':>11s}{'seed max':>11s}")
    for variant, geoms in sorted(res["variants"].items()):
        for g, s in sorted(geoms.items()):
            flag = "  <-- swamps signal" if s["swamps_signal"] else ""
            print(f"  {variant:14s}{g:10s}{s['floor_mae']:10.5f}"
                  f"{s['floor_over_prior'] * 100:11.1f}%{s['seed_min']:11.5f}"
                  f"{s['seed_max']:11.5f}{flag}")
    if res.get("_swamped"):
        print("\n  Cells whose floor swamps the expected signal (see exp_d output).")


def analyze_power(root: Path) -> None:
    res = _load(root / "power" / "power.json")
    rule("POWER ANALYSIS -> pre-registered n (§9)")
    if res is None:
        print("  not run")
        return
    print(f"  variant={res['variant']}  MDE={res['mde_frac_of_floor']:.0%} of floor  "
          f"alpha={res['alpha']}  power={res['power']}  assumed rho={res['assumed_rho']} "
          f"(conservative)")
    for g, v in res["per_geometry"].items():
        verdict(f"{g}: required n vs held-out pairs available",
                v["n_required"], v["n_test_available"],
                v["n_required"] <= v["n_test_available"])
    if res.get("_needs_extension"):
        print("\n  UNDERPOWERED -- extend the corpus with more shards before prereg.")


def analyze_e(root: Path, prereg_path: Path | None = None) -> None:
    """Sealed behind the §0.5 lock.  No skip flag exists."""
    out = root / "exp_e"
    artifacts = [out / "confound.npz", out / "results.json"]
    lock = prereg_lock.require("e", artifacts)   # raises PreregViolation on failure

    res = _load(out / "results.json")
    rule("EXPERIMENT E -- the confound (C1, C2, T5, T8) (§9-E)")
    print(lock.render())
    if res is None:
        print("\n  not run")
        return
    prereg = _read_prereg_thresholds()
    tau1 = prereg.get("tau_1", C.TAU_1_DEFAULT)
    prior = res["prior"]["pooled_mae_norm"]
    print(f"\n  prior={prior:.5f}   tau_1={tau1} (fraction of the s_std floor, "
          f"pre-declared before E)\n")

    variants = res["variants"]
    tests, labels = [], []

    for v, entry in sorted(variants.items()):
        floor = entry.get("floor_mae")
        print(f"  [{v}]  floor={floor if floor is None else f'{floor:.5f}'}")
        for cond, s in sorted(entry.get("conditions", {}).items()):
            print(f"    err({cond:9s}) = {s['point']:.5f} "
                  f"[{s['ci_low']:.5f},{s['ci_high']:.5f}]  "
                  f"{s['over_prior'] * 100:5.1f}% of prior")
        g = entry.get("G", {}).get("pooled")
        if g:
            eff = g.get("effect_over_floor")
            print(f"    G = {g['point']:+.5f} [{g['ci_low']:+.5f},{g['ci_high']:+.5f}]"
                  + (f"   = {eff:+.3f} x floor" if eff is not None else ""))

    # ---- C1 -----------------------------------------------------------
    std = variants.get("A-std", {})
    g_std = std.get("G", {}).get("pooled")
    if g_std and std.get("floor_mae"):
        ratio = g_std["point"] / std["floor_mae"]
        hi = g_std["ci_high"] / std["floor_mae"]
        lo = g_std["ci_low"] / std["floor_mae"]
        c1_supported = hi < tau1
        c1_refuted = lo > tau1
        print()
        verdict("C1  G(A-std)/floor  (supported if CI upper < tau_1)",
                ratio, tau1, c1_supported, ci=(lo, hi))
        if c1_refuted:
            print("    C1 REFUTED: the CI excludes tau_1 from BELOW.  Current practice is\n"
                  "    sound on this axis.  Report and stop (§11).  The surviving paper is\n"
                  "    C0 + the masking decomposition + the floors, as a characterisation.")
        elif not c1_supported:
            print("    INCONCLUSIVE: the CI straddles tau_1.  Report the continuous G and\n"
                  "    say so; do not move tau_1 (§0.1).")
        tests.append(g_std.get("p_one_sided", np.nan))
        labels.append("C1")

    # ---- C2 -----------------------------------------------------------
    dele = variants.get("A-del", {})
    if dele:
        ci = dele.get("conditions", {}).get("INTERACT")
        cd = dele.get("conditions", {}).get("DECOY")
        floor = dele.get("floor_mae")
        if ci and floor:
            mid = 0.5 * (floor + prior)
            verdict("C2  err(A-del, INTERACT) < midpoint(floor, prior)",
                    ci["point"], mid, ci["ci_high"] < mid,
                    ci=(ci["ci_low"], ci["ci_high"]))
        if cd:
            at_prior = cd["ci_low"] <= prior <= cd["ci_high"]
            verdict("C2  err(A-del, DECOY) ~ prior (construction check)",
                    cd["point"], prior, at_prior, ci=(cd["ci_low"], cd["ci_high"]))
            print("    This one is expected by construction: at s_del the arm is at\n"
                  "    canonical rest in every condition, so the DECOY frame pair contains\n"
                  "    literally zero information about the action.  It is a pipeline\n"
                  "    sanity check, not evidence (§1).")

    # ---- T5 -----------------------------------------------------------
    gt = variants.get("A-time", {}).get("G", {}).get("pooled")
    gd = variants.get("A-del", {}).get("G", {}).get("pooled")
    if g_std and gt and gd:
        d_std = abs(gt["point"] - g_std["point"])
        d_del = abs(gt["point"] - gd["point"])
        verdict("T5  |G(A-time)-G(A-std)| < |G(A-time)-G(A-del)|",
                d_std, d_del, d_std < d_del)
        print("    T5 cleared means the effect is driven by arm RETRACTION, not by\n"
              "    elapsed time.")

    # ---- T8 ablation ---------------------------------------------------
    gc = variants.get("A-clip-del", {}).get("G", {}).get("pooled")
    if g_std and gc and gd:
        closer_to_std = abs(gc["point"] - g_std["point"]) < abs(gc["point"] - gd["point"])
        verdict("T8  G(A-clip-del) closer to G(A-std) than to G(A-del)",
                gc["point"], g_std["point"], closer_to_std,
                ci=(gc["ci_low"], gc["ci_high"]))
        print("    Prediction: granting clip access re-opens the shortcut.  This turns\n"
              "    the would-be bug into a supporting result (§4-T8).")

    print("\n  Architecture B and all ablations are reported as descriptive support "
          "only (§10.2).")


def analyze_f(root: Path) -> None:
    res = _load(root / "exp_f" / "results.json")
    rule("EXPERIMENT F -- friction dose-response (§9-F)")
    if res is None:
        print("  not run")
        return
    for v, entry in sorted(res["variants"].items()):
        print(f"  [{v}]  slope(d MAE / d log friction) = {entry['log_friction_slope']:+.5f}  "
              f"spearman={entry['spearman_rho']:+.3f}  "
              f"{'monotone' if entry['monotone'] else 'NOT monotone'}")
        for m, s in entry["per_multiplier"].items():
            print(f"    x{m:<6s} MAE={s['point']:.5f} "
                  f"[{s['ci_low']:.5f},{s['ci_high']:.5f}]  "
                  f"{s['over_prior'] * 100:5.1f}% of prior")
    print("\n  Prediction: delayed-horizon error responds monotonically; standard-horizon\n"
          "  error stays flat.  The IDMs never trained on varied friction BY DESIGN --\n"
          "  detecting off-distribution dynamics is the metric's job (§9-F).")


def analyze_g(root: Path) -> None:
    res = _load(root / "exp_g" / "results.json")
    rule("EXPERIMENT G -- Lipschitz stratification (§9-G)")
    if res is None:
        print("  not run")
        return
    print(f"  L estimated from k={res['k']} sampled perturbations at eps={res['eps']}.")
    print(f"  {res['estimator_note']}")
    for g, v in res["per_geometry"].items():
        print(f"  [{g}] n={v['n']}  L_del quantiles(5/25/50/75/95) = "
              + ", ".join(f"{x:.2f}" for x in v["L_del_quantiles"]))
        for s in v["strata"]:
            if "G" in s:
                print(f"    stratum {s['stratum']} L in "
                      f"[{s['L_range'][0]:.2f},{s['L_range'][1]:.2f}]  "
                      f"n={s['n_pairs']:5d}  G={s['G']['point']:+.5f} "
                      f"[{s['G']['ci_low']:+.5f},{s['G']['ci_high']:+.5f}]")
    print("\n  Expect the shortcut worst in the LOW-sensitivity stratum and the fix to\n"
          "  matter most in the HIGH one.")


def analyze_masking(root: Path) -> None:
    res = _load(root / "exp_masking" / "results.json")
    rule("MASKING DECOMPOSITION (§8.4) -- Architecture B, descriptive support")
    if res is None:
        print("  not run")
        return
    for k, s in sorted(res["cells"].items()):
        v, mode = k.split("|")
        print(f"  {v:8s} ({s['horizon']:6s}) {mode:14s} MAE={s['point']:.5f} "
              f"[{s['ci_low']:.5f},{s['ci_high']:.5f}]  {s['over_prior'] * 100:5.1f}% of prior")
    print()
    for k, v in res["ordering_checks"].items():
        verdict(k[:52], v["full_over_prior"], "-", v["holds"])
        print(f"    {v['detail']}")
    print("\n  If these agree with the DECOY-based Experiment E result, T2 is dead: the\n"
          "  mask probes never remove the object from the SCENE, only from the INPUT.")


def analyze_h(root: Path) -> None:
    out = root / "exp_h"
    prereg_lock.require("h", [out / "results.json"])
    res = _load(out / "results.json")
    rule("EXPERIMENT H -- world-model ladder (C3) (§9-H)")
    if res is None:
        print("  not run")
        return
    lad = res.get("ladder_monotone")
    if lad is not None:
        verdict("ladder is a ladder: ground-truth error monotone across models 1-5",
                lad["spearman"], 0.9, lad["passed"])
        if not lad["passed"]:
            print("    The ladder FAILED.  Between-model claims are void (§9-H); fix the\n"
                  "    ladder before reporting C3(b).")
    for name, v in res.get("within_model", {}).items():
        verdict(f"C3(a) {name}: Spearman(standard rank, OG-AF rank)",
                v["spearman"], 1.0, v["spearman"] < 0.9)
    for name, v in res.get("between_model", {}).items():
        print(f"  C3(b) {name}: standard d={v['standard_effect']:+.3f}, "
              f"OG-AF d={v['ogaf_effect']:+.3f}, DINO d={v['dino_effect']:+.3f}")
    if "domain_gap" in res:
        print(f"\n  sim->generated feature distance: {res['domain_gap']}")


def _read_prereg_thresholds() -> dict:
    """Parse the thresholds frozen in prereg.md, if present."""
    p = prereg_lock.PREREG_PATH
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        s = line.strip()
        if s.startswith("- tau_1:"):
            try:
                out["tau_1"] = float(s.split(":", 1)[1].strip())
            except ValueError:
                pass
        if s.startswith("- eps_a:"):
            try:
                out["eps_a"] = float(s.split(":", 1)[1].strip())
            except ValueError:
                pass
    return out


ANALYSES = {
    "0": analyze_0, "a": analyze_a, "b": analyze_b, "c": analyze_c,
    "corpus": analyze_corpus, "d": analyze_d, "power": analyze_power,
    "e": analyze_e, "f": analyze_f, "g": analyze_g,
    "masking": analyze_masking, "h": analyze_h,
}
UNSEALED_ORDER = ["0", "a", "b", "c", "corpus", "d", "power", "masking"]
SEALED_ORDER = ["e", "f", "g", "h"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="all",
                    choices=sorted(ANALYSES) + ["all", "unsealed"])
    ap.add_argument("--results", type=Path, default=C.RESULTS_ROOT)
    args = ap.parse_args()

    if args.exp == "unsealed":
        keys = UNSEALED_ORDER
    elif args.exp == "all":
        keys = UNSEALED_ORDER + SEALED_ORDER
    else:
        keys = [args.exp]

    failures = []
    for k in keys:
        try:
            ANALYSES[k](args.results)
        except prereg_lock.PreregViolation as exc:
            rule(f"EXPERIMENT {k.upper()} -- SEALED")
            print(str(exc))
            failures.append(k)
    rule("")
    if failures:
        print(f"sealed and not analysed: {', '.join(failures)} "
              f"(write, commit and PUSH prereg.md first -- §11, §3.7, §0.5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
