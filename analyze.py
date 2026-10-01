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
import re
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


def _holm_report(out: Path, variants: dict, tau1: float, prior: float) -> None:
    """Holm-corrected one-sided bootstrap p-values for the confirmatory family.

    prereg §10.2 promises Holm across the confirmatory tests; until now the
    p-values were never computed (``tests``/``labels`` were collected and
    dropped) and stats.holm_correction was dead code.  They are computed
    HERE -- the only place arrays become verdicts -- from the per-sample
    arrays exp_e wrote, against the SAME nulls the CI verdicts above use:
      C1: G(A-std)            < tau_1 * floor(A-std)
      C2: err(A-del, INTERACT) < midpoint(floor(A-del), prior)
    T5 compares two point estimates with no registered null and stays
    descriptive, so the family is {C1, C2}.  Never lets a failure here take
    the verdict table down with it.
    """
    npz = out / "confound.npz"
    if not npz.exists():
        print(f"\n  Holm: {npz} missing; p-values not computed")
        return
    try:
        z = np.load(npz, allow_pickle=False)
        pvals, labels = [], []

        std = variants.get("A-std", {})
        if _finite_positive(std.get("floor_mae")):
            vals, ids = {}, {}
            for k in z.files:
                m = re.match(r"^A-std_seed(\d+)_G_paired$", k)
                if m:
                    s = int(m.group(1))
                    vals[s], ids[s] = z[k], z[f"A-std_seed{s}_G_pair_ids"]
            if vals:
                pd = stats.PairedData(values=vals, pair_ids=ids)
                pvals.append(stats.one_sided_p(pd, tau1 * float(std["floor_mae"]), "less",
                                               n_boot=C.N_BOOTSTRAP, rng_seed=C.BOOTSTRAP_SEED))
                labels.append("C1: G(A-std) < tau_1 * floor")

        dele = variants.get("A-del", {})
        if _finite_positive(dele.get("floor_mae")):
            vals, ids = {}, {}
            for k in z.files:
                m = re.match(r"^A-del_seed(\d+)_INTERACT_mae$", k)
                if m:
                    s = int(m.group(1))
                    vals[s] = z[k]
                    ids[s] = stats.pair_id(z[f"A-del_seed{s}_INTERACT_geometry"],
                                           z[f"A-del_seed{s}_INTERACT_tuple"])
            if vals:
                pd = stats.PairedData(values=vals, pair_ids=ids)
                mid = 0.5 * (float(dele["floor_mae"]) + prior)
                pvals.append(stats.one_sided_p(pd, mid, "less",
                                               n_boot=C.N_BOOTSTRAP, rng_seed=C.BOOTSTRAP_SEED))
                labels.append("C2: err(A-del, INTERACT) < midpoint")

        if not pvals:
            print("\n  Holm: no confirmatory p-values available (floors missing?)")
            return
        adj = stats.holm_correction(pvals, labels)
        print(f"\n  Holm-Bonferroni over the confirmatory family ({len(pvals)} tests; "
              "T5 descriptive), one-sided bootstrap p:")
        for lab in labels:
            print(f"    {lab:44s} p={adj[lab]['p_raw']:.4f}   "
                  f"Holm-adjusted={adj[lab]['p_holm']:.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n  Holm: p-values not computed ({type(exc).__name__}: {exc})")


def _finite_positive(x) -> bool:
    """json.dump writes NaN, json.loads accepts it, and NaN is truthy -- so a
    missing floor must be tested explicitly, not with ``if floor:``."""
    try:
        return x is not None and float(x) == float(x) and float(x) > 0
    except (TypeError, ValueError):
        return False


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
    if g_std and not _finite_positive(std.get("floor_mae")):
        print("\n  C1: floor_mae for A-std is missing/NaN -- run exp_d_floors.py and "
              "re-run exp_e_confound.py; no verdict without the floor")
    if g_std and _finite_positive(std.get("floor_mae")):
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

    # ---- C2 -----------------------------------------------------------
    dele = variants.get("A-del", {})
    if dele:
        ci = dele.get("conditions", {}).get("INTERACT")
        cd = dele.get("conditions", {}).get("DECOY")
        floor = dele.get("floor_mae")
        if ci and not _finite_positive(floor):
            print("  C2: floor_mae for A-del is missing/NaN -- run exp_d_floors.py first")
        if ci and _finite_positive(floor):
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

    # ---- Holm across the confirmatory family ----------------------------
    _holm_report(out, variants, tau1, prior)

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


def analyze_subset(root: Path) -> None:
    """Descriptive: what a manipulator-only verification subset costs (§9-0b)."""
    res = _load(root / "verification_subset" / "results.json")
    rule("VERIFICATION SUBSET -- arXiv:2604.01985 Prop. 3.1 conditions, descriptive")
    if res is None:
        print("  not run")
        return
    summ = res["_summary"]
    print(f"  {'geometry':10s} {'horizon':8s} {'(i) scene-indep':>16s} "
          f"{'(iii) subset':>13s} {'object-only':>12s}")
    for r in summ["rows"]:
        print(f"  {r['geometry']:10s} {r['horizon']:8s} {str(r['cond_i']):>16s} "
              f"{r['subset_over_prior'] * 100:12.1f}% "
              f"{r['object_only_over_prior'] * 100:11.1f}%")
    both = summ["horizons_where_both_hold"]
    print(f"\n  Both (i) and (iii) hold at: {', '.join(both) if both else 'no horizon'}")
    print("\n  Read: the conditions that make a manipulator-only subset a sound\n"
          "  VERIFIER are satisfied exactly at the horizon the standard metric uses,\n"
          "  and fail at the OG-AF horizon.  Condition (i) IS object-blindness, so a\n"
          "  score built on that subset has G = 0 by construction.  The guarantee and\n"
          "  the confound are the same property read for different purposes.")
    print(f"\n  (ii) {summ['condition_ii_status']}")


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
        # Disagreement is the prediction: pass when the two rankings correlate
        # BELOW 0.9 (the threshold printed is the one applied).
        verdict(f"C3(a) {name}: Spearman(standard rank, OG-AF rank) < 0.9",
                v["spearman"], 0.9, v["spearman"] < 0.9)
    for name, v in res.get("between_model", {}).items():
        if not isinstance(v, dict):          # e.g. {"void": "ladder not monotone"}
            print(f"  C3(b) {name}: {v}")
            continue
        if name.endswith("_paired"):
            cells = ", ".join(f"{k}={p['d_paired']:+.3f} [{p['ci_low']:+.3f},{p['ci_high']:+.3f}]"
                              for k, p in v.items() if isinstance(p, dict) and "d_paired" in p)
            print(f"  C3(b) {name} (same tuples, n={v.get('ogaf', {}).get('n_pairs', '?')}): {cells}")
            continue
        print(f"  C3(b) {name}: standard d={v['standard_effect']:+.3f}, "
              f"OG-AF d={v['ogaf_effect']:+.3f}, DINO d={v['dino_effect']:+.3f}")
    if res.get("domain_shift"):
        print("\n  domain shift (INTERACT generations): DINO(gen, real), clean-vs-augmented IDM delta")
        for n, v in res["domain_shift"].items():
            print(f"    {n:24s} dino={v['dino_generated_vs_real']:.5f}  "
                  f"std_aug={v['standard_augmented']:.5f}  std_clean={v['standard_clean']:.5f}  "
                  f"delta={v['clean_minus_augmented']:+.5f}")
    if res.get("domain_gap"):
        print("\n  appearance-gap control (ABSENT: no object, physics divergence = 0 by construction)")
        for n, v in res["domain_gap"].items():
            if "missing" in v:
                print(f"    {n:24s} {v['missing']}")
                continue
            print(f"    {n:24s} n={v['n']:4d} standard={v['standard']:.5f}  ogaf={v['ogaf']:.5f}  "
                  f"dino={v['dino']:.5f}  gt={v['ground_truth']:.5f}")


_PREREG_NUMBER = r"\s*[:=]\s*(?:<<)?\s*([0-9]*\.?[0-9]+)"


def _read_prereg_thresholds() -> dict:
    """Parse the thresholds frozen in prereg.md, if present.

    The template writes them as ``- τ₁ (...): `- tau_1: <<0.25>>` `` -- the
    ``tau_1:`` token is mid-line, inside backticks, and may still be wrapped in
    the ``<<>>`` placeholder markers.  A line-prefix match therefore never
    fired and analysis silently fell back to config.TAU_1_DEFAULT, which the
    lock does not hash.  Prefer a dedicated ``- tau_1:`` line; otherwise take
    the first ``tau_1: <number>`` anywhere in the file.
    """
    p = prereg_lock.PREREG_PATH
    if not p.exists():
        return {}
    text = p.read_text()
    out = {}
    for key in ("tau_1", "eps_a"):
        m = (re.search(rf"^\s*-\s*{key}{_PREREG_NUMBER}", text, re.MULTILINE)
             or re.search(rf"{key}{_PREREG_NUMBER}", text))
        if m:
            out[key] = float(m.group(1))
    return out


ANALYSES = {
    "0": analyze_0, "a": analyze_a, "b": analyze_b, "c": analyze_c,
    "corpus": analyze_corpus, "d": analyze_d, "power": analyze_power,
    "e": analyze_e, "f": analyze_f, "g": analyze_g,
    "masking": analyze_masking, "subset": analyze_subset, "h": analyze_h,
}
UNSEALED_ORDER = ["0", "a", "b", "c", "subset", "corpus", "d", "power", "masking"]
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
