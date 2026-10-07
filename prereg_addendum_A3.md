# Pre-registration addendum A3: compliant arm, mild corruptions, arm-masked baseline

Registered: `2026-10-07` · Repository: `https://github.com/Kushaan-N/embodiment-shortcut` · Parent registrations: `prereg.md` (2026-09-29), `prereg_addendum_H2.md` (2026-10-05) · Commit: filled by git

Registered, committed and pushed **before** the compliant-arm calibration,
before any corpus, IDM or world model for these experiments exists, and before
any statistic below is computed. It answers three objections raised in review
of the workshop draft. It changes no earlier result. `prereg_lock.check(
prereg_path=prereg_addendum_A3.md)` must PASS before `analyze.py --exp a3`
reports, and before `wm/exp_h.py --ladder wm/ladder_v3.yaml` writes a result.

---

## K. Compliant arm: does the shortcut survive when contact moves the arm?

**Objection.** The main study clamps the arm to its command, so physics cannot
move it and the standard metric's flatness is partly built in.

**K.1 Gain, chosen by rule (calibration, no IDM).** Position-actuated arm
(`OGAF_ARM_CONTROL=position`), candidate gains kp ∈ {40000, 10000, 2500, 625},
60 INTERACT tuples per geometry, no rendering. Deflection = the T9 value: max
over the rollout of the arm's distance from the same command executed with
arm–object contact disabled. Select the **stiffest** gain whose median
deflection is at least the pooled `delta_pos_min` (one video frame of object
motion, Experiment A) in every geometry, with at least 80 % of rollouts passing
every other validator. If none qualifies, K stops and reports that. Output:
`results/exp_k_calibration/results.json` (`calibrate_compliance.py`).

**K.2 Pipeline.** At the selected gain, with T9 recorded but not excluding
(`OGAF_T9_EXCLUDES=0`): the full corpus (2000 tuples × 3 geometries × 3
conditions, both clip variants), the corpus gate with a 20 % per-validator
exclusion cap (matching K.1's keep rule), frame stores, A-std and A-del IDMs
(5 seeds each, identical protocol), Experiment D floors and Experiment E, in a
separate data root.

**K1 (confirmatory).** The main study's C1 replicated under compliance:
G(A-std)/reference, 95 % hierarchical-bootstrap CI, threshold τ₁ = 0.25 as
registered in `prereg.md`.
- **Supported** (the shortcut survives compliance) iff the CI upper bound < 0.25.
- **Refuted** (compliance removes the shortcut at this threshold) iff the CI
  lower bound > 0.25.
- **Inconclusive** otherwise.

Either outcome is reported; a refutation scopes the paper's claim to stiff
position control. **Descriptive:** object share = (err DECOY − err INTERACT) /
(0.25 − err INTERACT) for A-std and A-del, against 0.003 and 0.98 in the
clamped study.

## M. Milder physics corruptions for the world-model test

**Objection.** H2's corruption (friction ×0.1) is gross: OG-AF scores that
model near chance.

**Models.** Two world models trained exactly like WM-base (120 000 steps, same
architecture, data size and seed) on full INTERACT corpora at friction **×0.25**
and **×0.5**. Both lose no rollouts to the validators at calibration (keep 1.00
at ×0.25 in `results/exp_h2_calibration`), so there is no data-size confound.
Generation, validation and scoring are as in H2: 193 held-out box tuples,
augmented A-std at the s_std frame, augmented A-del at the last frame, IDM seed
0, plus the ABSENT control. Spec: `wm/ladder_v3.yaml`.

**Per-model validity gate.** The corrupted model must be measurably worse in
physics than WM-base-100: paired mean difference in object-region error
(`prereg_addendum_H2.md` §4), corrupted minus base, with bootstrap CI lower
bound > 0. If the gate fails, that model's test is void, because the
corruption was too mild to change the generated physics.

**M1 (confirmatory, one test per model).** The decision rule of
`prereg_addendum_H2.md` §5 (paired d for both metrics and their contrast Δ on
the same resamples), with **97.5 %** CIs (Bonferroni over the two models).
Supported iff d_ogaf's CI and Δ's CI both lie above 0; refuted iff Δ's CI lies
below 0; otherwise inconclusive.

## N. Arm-masked standard-horizon baseline (descriptive)

**Objection.** An obvious alternative to OG-AF is to keep the standard horizon
and hide the arm.

**Definition, fixed here.** `A-std-armmask`: Architecture A on the pair
(s_0, s_std), with the arm's pixels in both frames replaced by the empty-scene
render using the simulator's segmentation, which is the §8.4 masking already
used for the probes. Same protocol, 5 seeds, trained on all three conditions.
Reported in Experiment E (G/reference, object share) and in the
error-versus-displacement analysis next to A-std and A-del. **Prediction**
(descriptive, not a hypothesis test): its object share lies far above A-std's.
On generated video it is not scored, because generated frames carry no
segmentation.

## Who fixed these values

The researcher asked for these experiments (2026-10-07) and delegated their
design to the implementing agent (Claude), which fixed every value above before
any calibration, corpus, training or statistic for them existed. The gain
candidates, the deflection criterion, the ×0.25 and ×0.5 corruptions, the 97.5 %
level and the masked variant were chosen from the review's objections and the
already-public H2 calibration record only.
