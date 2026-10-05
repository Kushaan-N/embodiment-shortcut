# Pre-registration addendum H2: a world-model ladder that is a ladder

Registered: `2026-10-05` · Repository: `https://github.com/Kushaan-N/embodiment-shortcut` · Parent registration: `prereg.md` (commit d2fdc9e, 2026-09-29) · Commit: filled by git

This addendum is registered, committed and pushed **before** the calibration
in §2 is run, before the corrupted world model in §3 is trained, and before any
H2 score exists. It does not alter `prereg.md` or the Experiment H result
(C3(a) PASS ×5; ladder not monotone, Spearman 0.70 < 0.9; C3(b) void), which
stands and is reported as registered. H2 is a separate, additional test.
`prereg_lock.check(prereg_path=prereg_addendum_H2.md)` must PASS (committed,
byte-identical, pushed to the public remote, earlier than every H2 artifact)
before `wm/exp_h.py --ladder wm/ladder_v2.yaml` writes a result.

## 1. Why Experiment H's ladder failed (diagnosis, stated before H2 data)

The ladder's validity gate measured ground truth as whole-clip pixel error.
The object covers about 2 % of a 224×224 frame, so that number is dominated by
how well the arm and background are rendered, not by where the object goes:
the x3-friction model (0.00402) landed beside the 10 %-trained base model
(0.00428), below the data-poor model (0.00704). x3 friction was also never
checked to be a large physical change. Both are fixed below.

## 2. Corruption, chosen by a mechanical rule (ground truth only)

- Candidates: object friction multiplier k in {0.1, 0.25, 4, 10}.
- For each k, simulate the box INTERACT corpus shards 0-1 (200 tuples, all
  three geometries for the exclusion check) at friction ×k and render the WM
  video clips exactly as for training.
- **Physics divergence D(k)** = mean over box tuples valid at both ×1 and ×k
  of the object-region error (§4) between the ×k real clip and the ×1 real
  clip -- the score a *perfect* world model of the wrong physics would get.
- **Threshold** T = 1.25 × the object-region error (§4) of the existing
  `WM-data-poor` generations (193 held-out box tuples), so that even a perfect
  wrong-physics model is worse than the weakest correct-physics model.
- **Exclusion check**: at ×k, at least 80 % of rollouts pass every §6
  validator in each geometry (shards 0-1).
- **Selection**: among candidates with D(k) ≥ T that pass the exclusion
  check, the one with the smallest |ln k| (the mildest qualifying
  corruption); ties go to the smaller k.
- If no candidate qualifies, H2 stops and reports that a friction corruption
  cannot produce a valid ladder in this setting.
- The calibration writes `results/exp_h2_calibration/results.json`, and the
  selected k is copied into `wm/ladder_v2.yaml` before training.

## 3. Ladder v2

Models 1-4 are the existing, already-trained checkpoints (WM-base-100, -30,
-10, WM-data-poor; same generations, same 193 held-out box tuples).
Model 5, `WM-physics-corrupted-v2`, is trained exactly like WM-base
(120 000 steps, same architecture, data size and seed) on the full ×k INTERACT
corpus (2000 tuples per geometry). Generation, per-item validation and
scoring are unchanged from Experiment H (augmented A-std at the s_std frame,
augmented A-del at the last frame, IDM seed 0, DINO distance at s_del).

## 4. Ground truth for the ladder gate: object-region error

For a generated clip G and the real ×1 INTERACT clip R of the same tuple, with
the real ×1 ABSENT clip A (same arm, no object), the object mask of frame t is
M_t = { pixels where max_channel |R_t − A_t| > 20 } (object and its shadow).
Object-region error = Σ_t Σ_{M_t} |G_t − R_t| / (3 · Σ_t |M_t|) / 255.

**Ladder gate (H2)**: Spearman(model rank 1..5, mean object-region error)
> 0.9, the same threshold as Experiment H. If it fails, C3(b)-H2 is void.
Whole-clip error is still reported.

## 5. C3(b)-H2: decision rule

On the 193 paired held-out tuples, per-rollout IDM error of model 5 minus
model 1 (WM-base-100), for the standard metric (A-std) and OG-AF (A-del):
paired effect d = mean(diff)/sd(diff); joint percentile bootstrap over tuples
(10 000 replicates, seed 987654321), giving 95 % CIs for d_std, d_ogaf and the
contrast Δ = d_ogaf − d_std on the same resamples.

- **Supported** iff the gate passes, d_ogaf's CI lies above 0, and Δ's CI lies
  above 0 (OG-AF separates the wrong-physics model, and separates it more than
  the standard metric does).
- **Refuted** iff the gate passes and Δ's CI lies below 0.
- **Inconclusive** otherwise. Report every d and its CI either way.

Single confirmatory test in this addendum; no multiplicity correction.
Descriptive: DINO distance, unpaired d, the ABSENT appearance-gap control for
model 5, whole-clip ground truth.

## 6. Who fixed these values

The researcher asked for the ladder to be rebuilt and registered first
(2026-10-05) and delegated the design to the implementing agent (Claude),
which fixed every value above before any calibration, training or H2 score
existed, using only the diagnosis in §1 (from the already-public Experiment H
record).
