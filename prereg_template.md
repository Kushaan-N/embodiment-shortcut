# TEMPLATE -- copy to `prereg.md`, fill EVERY `<<...>>`, commit, and PUSH

Do not commit this template as `prereg.md` with placeholders intact.
`analyze.py --exp e` will run against whatever is committed, and an unfilled
threshold is a threshold chosen after the fact.

**When:** after Experiment D and `power.py`, and **before any per-condition
Experiment E statistic is computed** (§3.7, §11).

**How:**

```bash
cp prereg_template.md prereg.md
$EDITOR prereg.md                     # fill every <<...>> from A/C/D/power outputs
git add prereg.md && git commit -m "pre-registration (frozen before Experiment E)"
git push                              # PUBLIC timestamp; a local commit is rewritable
python prereg_lock.py                 # must report PASS before exp_e_confound.py runs
```

Optionally also file the same document as an OSF registration -- ten minutes for
an external timestamp nobody in this subfield has (§11).

---

# Pre-registration: the embodiment shortcut in action-following metrics

Registered: `<<YYYY-MM-DD>>`  ·  Repository: `<<public URL>>`  ·  Commit: filled by git

## 0. Status of inputs

| Input | Value | Source |
|---|---|---|
| Gate C0 | `<<PASS/FAIL>>` | `results/exp_0/results.json` |
| Gate B | `<<PASS/FAIL>>`, winning encoder/pooling `<<...>>` | `results/exp_b/results.json` |
| Gate C | `<<PASS/FAIL>>` | `results/exp_c/results.json` |
| Corpus gate (T11, T7) | `<<PASS/FAIL>>`, max R² `<<...>>` | `results/validate_corpus/report.json` |
| Experiment D floors | see §4 | `results/exp_d/floors.json` |
| Power analysis | see §5 | `results/power/power.json` |

## 1. Hypothesis

The standard action-following metric is dominated by **embodiment rendering**,
not object physics. An IDM minimising action-prediction error learns to read the
manipulator and largely ignore the object -- not as a pathology, but because
that is the optimal solution to the task as posed.

Primary estimator, on held-out data, paired by `(action, seed)`:

```
G = mean_p [ err(DECOY_p) - err(INTERACT_p) ]
```

computed as the **mean of per-pair differences**, never as a difference of
condition means.

## 2. Frozen definitions

- Horizons: `s_0` = index 0; `s_std` = `<<index>>` (last step of the push);
  `s_del` = `max(settled_at, arm_rest_at)`; `s_time` = same index as `s_del`
  with the ε-withdrawn arm.
- ε withdrawal: `<<...>>` m, over `<<...>>` steps, shared by both arm variants.
- Settling thresholds (Experiment A, never hardcoded):
  box `<<...>>` m/s, sphere `<<...>>` m/s, cylinder `<<...>>` m/s.
- `delta_pos_min` / `delta_rot_min` per geometry: `<<...>>`.
- ε_a (working action separation, Experiment C): `- eps_a: <<0.10>>`
- Split unit: `(geometry, tuple_index)`; fractions 80/10/10; salt `<<...>>`.
- Inclusion policy: a rollout is included iff **every** §6 validator passes.
  Exclusions are counted and reported per (geometry, condition, reason).
  Current measured exclusion rates: `<<...>>`.
- n per cell (from §5): `<<...>>`.
- Primary geometry/pooling for the confirmatory tests: **pooled over geometries**.
- Bootstrap: hierarchical (resample `(action, seed)` tuples within seed, then
  resample seeds), 10,000 replicates, percentile 95% CIs, seed `<<...>>`.
  Seed-level min/max reported alongside.
- All hypothesis tests **one-sided** in the direction stated below.
- τ₁ (C1 threshold, as a fraction of the s_std floor): `- tau_1: <<0.25>>`
  This fraction is a convention, chosen before Experiment E; the continuous G
  is reported so readers can apply their own.

## 3. Confirmatory tests (Architecture A only, pooled geometry, Holm-corrected across the three)

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
- *Cleared* if `|G(A-time) − G(A-std)| < |G(A-time) − G(A-del)|`.

## 4. Floors (Experiment D, INTERACT held-out only)

| Variant | Geometry | Floor (norm. MAE) | % of prior |
|---|---|---|---|
| A-std | pooled | `<<...>>` | `<<...>>` |
| A-del | pooled | `<<...>>` | `<<...>>` |
| A-time | pooled | `<<...>>` | `<<...>>` |
| ... | ... | | |

Prior baseline: normalised MAE = 0.25, MSE = 1/12 (analytic, uniform actions);
empirical constant-predictor fit on train = `<<...>>` (these must agree).

## 5. Power

MDE = `<<0.25>>` × the A-std s_std floor = `<<...>>`; one-sided α = 0.05;
power = 0.90; assumed pair correlation ρ = 0 (conservative -- pairing can only
help). Required n per geometry: `<<...>>`. Pre-registered
`n = min(2000, computed)` = `<<...>>`.

## 6. Exploratory / descriptive (reported, never the basis of a claim)

- Architecture B (all variants, 10 seeds).
- The masking decomposition (§8.4), with its predicted orderings.
- A-clip-del (T8 ablation): prediction `G(A-clip-del) ≈ G(A-std)`.
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

- `<<none as of registration>>`
