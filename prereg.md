# Pre-registration: the embodiment shortcut in action-following metrics

Registered: `2026-09-29`  ·  Repository: `https://github.com/Kushaan-N/embodiment-shortcut`  ·  Commit: filled by git

## 0. Status of inputs

| Input | Value | Source |
|---|---|---|
| Gate C0 | PASS (original, 3 seeds) and PASS (Unity re-run) | `results/exp_0/results.json`, `results/exp_0_repro/results.json` |
| Gate B | PASS, winning encoder/pooling `facebook/dinov3-vitl16-pretrain-lvd1689m` / CLS (+13.90 σ translation; rotation −1.50 σ, fails for every encoder) | `results/exp_b/results.json` |
| Gate C | PASS (unidentifiable at s_del: box 0.0 %, sphere 2.3 %, cylinder 3.4 % vs 25 %) | `results/exp_c/results.json` |
| Corpus gate (T11, T7, completeness, clips) | PASS, max R² 0.00000 | `results/validate_corpus/report.json` |
| Experiment D floors | see §4 | `results/exp_d/floors.json` |
| Power analysis | see §5 | `results/power/power.json` |

## 1. Hypothesis

The standard action-following metric is dominated by **embodiment rendering**,
not object physics. An IDM minimising action-prediction error learns to read the
manipulator and largely ignore the object -- not as a pathology, but because
that is the optimal solution to the task as posed.

Primary estimator, on held-out data, paired by `(geometry, action, seed)`:

```
G = mean_p [ err(DECOY_p) - err(INTERACT_p) ]
```

computed as the **mean of per-pair differences**, never as a difference of
condition means.

## 2. Frozen definitions

- Horizons: `s_0` = index 0; `s_std` = 474 (last step of the push, 500 Hz);
  `s_del` = `max(settled_at, arm_rest_at)` (arm_rest_at = 749); `s_time` = same
  index as `s_del` with the ε-withdrawn arm.
- ε withdrawal: 0.0015 m, over 10 steps (0.02 s), shared by both arm variants.
- Settling thresholds (Experiment A, never hardcoded): linear speed
  box 0.0952 m/s, sphere 0.0903 m/s, cylinder 0.1074 m/s
  (angular: box 0.0493, sphere 0.5934, cylinder 0.0266 rad/s).
- `delta_pos_min` per geometry: box 4.76 mm, sphere 4.52 mm,
  cylinder 5.37 mm. `delta_rot_min`: box 0.00247 rad, cylinder
  0.00036 rad, sphere n/a (rotation-invariant). Measured at the video
  frame rate (20 Hz, stride 25).
- ε_a (working action separation, Experiment C): `- eps_a: 0.10`
- Split unit: `(geometry, tuple_index)`; fractions 80/10/10; salt `ogaf-v3-split`.
- Inclusion policy: a rollout is included iff **every** §6 validator passes.
  Exclusions are counted and reported per (geometry, condition, reason).
  Current measured exclusion rates: box/ABSENT 0.00%, box/DECOY 0.00%, box/INTERACT 0.00%, cylinder/ABSENT 0.00%, cylinder/DECOY 0.05%, cylinder/INTERACT 1.10%, sphere/ABSENT 0.00%, sphere/DECOY 0.00%, sphere/INTERACT 0.00%.
- n per cell (from §5): required 102 (box), 84 (sphere), 89 (cylinder), 96
  pooled; every held-out test pair is used (~190-230 per geometry, 611 pooled
  INTERACT after exclusions), which exceeds the requirement.
- Primary geometry/pooling for the confirmatory tests: **pooled over geometries**.
- Bootstrap: hierarchical (resample `(action, seed)` tuples within seed, then
  resample seeds), 10,000 replicates, percentile 95% CIs, seed 987654321.
  Seed-level min/max reported alongside.
- All hypothesis tests **one-sided** in the direction stated below.
- τ₁ (C1 threshold, as a fraction of the s_std floor): `- tau_1: 0.25`
  This fraction is a convention, chosen before Experiment E; the continuous G
  is reported so readers can apply their own.

## 3. Confirmatory tests (Architecture A only, pooled geometry, Holm-corrected across C1 and C2)

**C1 -- the standard metric is largely insensitive to object dynamics.**
- *Supported* if `G(A-std)/floor < τ₁` with the one-sided 95% CI excluding τ₁
  from above.
- *Refuted* if `G(A-std)/floor` is substantially positive, i.e. the CI excludes
  τ₁ from below. **Then current practice is sound on this axis: report and
  stop.** The surviving paper is C0 + the masking decomposition + the floors,
  as a characterisation of the metric. This branch is written out in advance
  and is as publishable internally as the positive one (§0.2).
- *Inconclusive* if the CI straddles τ₁. Report the continuous G and say so.
  Do not move τ₁.

**C2 -- object dynamics alone suffice once the manipulator is at rest.**
- *Supported* if `err(A-del, INTERACT) < midpoint(floor, prior)` with CI,
  **and** `err(A-del, DECOY)` lies within CI of the prior.
- The DECOY half is a **pipeline sanity check, expected by construction**: at
  `s_del` the arm is at canonical rest in every condition, so the DECOY frame
  pair contains zero information about the action. It is reported as such and
  is not offered as evidence.

**T5 -- the effect is retraction, not elapsed time.**
- *Cleared* if `|G(A-time) − G(A-std)| < |G(A-time) − G(A-del)|`. A comparison
  of point estimates with no registered null: reported descriptively and not
  part of the Holm family.

Holm-Bonferroni is applied to the one-sided bootstrap p-values of C1 and C2,
computed against the same nulls as the CI verdicts (`analyze.py`).

## 4. Floors (Experiment D, INTERACT held-out only)

| Variant | Geometry | Floor (norm. MAE) | % of prior |
|---|---|---|---|
| A-std | pooled | 0.00482 | 1.9% |
| A-del | pooled | 0.11171 | 44.7% |
| A-time | pooled | 0.01026 | 4.1% |
| A-clip-del | pooled | 0.00556 | 2.2% |
| A-std | box | 0.00552 | 2.2% |
| A-std | sphere | 0.00443 | 1.8% |
| A-std | cylinder | 0.00456 | 1.8% |
| A-del | box | 0.11718 | 46.9% |
| A-del | sphere | 0.11368 | 45.5% |
| A-del | cylinder | 0.10374 | 41.5% |

Prior baseline: normalised MAE = 0.25, MSE = 1/12 (analytic, uniform actions);
empirical constant-predictor MAE on the Experiment 0 test split: box
0.2444, sphere 0.2449, cylinder 0.2477 (within ~2 % of
the analytic value).

## 5. Power

MDE = 0.25 × the A-std s_std floor = 0.00120 (pooled; box 0.00138,
sphere 0.00111, cylinder 0.00114); one-sided α = 0.05; power = 0.90;
assumed pair correlation ρ = 0 (conservative -- pairing can only help).
Required n per geometry: box 102, sphere 84, cylinder 89 (pooled 96).
Pre-registered `n = min(2000, computed)` = 102 / 84 / 89; no corpus extension needed.

## 6. Exploratory / descriptive (reported, never the basis of a claim)

- Architecture B (all variants, 10 seeds).
- The masking decomposition (§8.4), with its predicted orderings.
- A-clip-del (T8 ablation): prediction `G(A-clip-del) ≈ G(A-std)`. (Its floor,
  2.2 % of prior, already sits beside A-std's 1.9 %.)
- ABSENT: quantifies the residual OOD component of any INTERACT/ABSENT gap (T2).
- Per-geometry breakdowns.
- Experiments F and G.

## 7. What would make us abandon the claim

- C1 refuted as defined in §3.
- Gate B fails for every encoder/pooling: the finding becomes "DINO-family
  features cannot resolve contact-scale geometry", and the project redirects.
- Gate C fails: too much of the action space is unidentifiable at `s_del` and
  OG-AF would punish correct models.
- The corpus gate fails (T11 leak or T7 split leak).
- Experiment H's ladder is not monotone under privileged ground-truth error:
  between-model C3(b) claims are then void.

## 8. Deviations

Any deviation from this document after registration is recorded here with its
date and reason, and the pre-registered analysis is reported alongside the
deviating one.

- none as of registration

## 9. Who fixed these values

The researcher (Kushaan Naskar) delegated the open registration decisions to
the implementing agent (Claude) on 2026-09-29, and the agent took the
template's pre-declared defaults without reference to any Experiment E
statistic, none of which existed: τ₁ = 0.25; MDE = 0.25 × the A-std floor;
Holm family {C1, C2} (T5 has no registered null and is descriptive); both
Experiment 0 records cited; the 13 implementing-agent protocol fields
(`python protocol.py`) frozen as they stand.  Architecture B / the masking
decomposition were running at registration time; they are exploratory (§6).
