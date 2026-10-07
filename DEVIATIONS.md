# Deviations from the pre-registration

`prereg.md` §8 says deviations are recorded in that file, but `prereg_lock.py`
fails if `prereg.md` changes after its registering commit (d2fdc9e,
2026-09-29). They are recorded here instead. Each entry gives the date, what
changed, why, and whether any **confirmatory** result (C1, C2) moved. The
pre-registered analysis is reported alongside any deviating one.

## 2026-09-29 — Experiment F: x0.25 and x0.5 shared one frame store (bug fix)

- **What:** `idm_data._store_paths` built names with `Path.with_suffix`, which
  read the `.25` of `…_fm0.25` as an extension, so the x0.25 and x0.5 stores
  were one file (`…_fm0.frames.npy`). The first Experiment F run scored both
  multipliers on the x0.5 data (identical MAE and CI at both).
- **Fix:** commit e21e1d1 (names by concatenation) + regression test c4d48a6;
  the two stores were rebuilt and F re-scored.
- **Confirmatory impact: none.** Only fractional multipliers were affected. The
  baseline corpus (x1), Experiments D/E, and the x2/x3/x4 corpora use integer
  names and were verified unaffected.

## 2026-09-29 — Experiment G: sphere/cylinder strata were empty (bug fix)

- **What:** strata were matched against Experiment E's pair ids using raw tuple
  indices; E's ids are `stats.pair_id` (geometry-encoded), which only coincide
  for box. Sphere and cylinder got no stratified G.
- **Fix:** commit af7b732; G re-run. **Confirmatory impact: none** (G is §6
  exploratory).

## 2026-09-29 — Two exploratory predictions were mis-specified (reported, NOT changed)

The predictions below are left exactly as registered and their FAIL verdicts
stand. The notes say what the data show; neither is offered as a claim.

- **Experiment F, "delayed-horizon error responds monotonically".** The A-del
  (OG-AF) IDM was trained at x1, so off-distribution friction in *either*
  direction raises its error: 61.2 / 47.8 / **44.7** / 47.5 / 50.2 % of prior at
  x0.25 / 0.5 / 1 / 2 / 4 — U-shaped, not monotone. The standard-horizon A-std
  stays at 1.8–1.9 % throughout, even though the object's settled position
  differs by up to 4 cm between x0.25 and x0.5 (the arm is kinematically
  identical, and A-std reads the arm).
- **§8.4 masking orderings.** At `s_std`, arm pixels alone match the full frame
  (9.9 vs 11.7 % of prior), but the clause "object pixels ≈ prior" is too strong:
  the pushed object alone reaches 48 %. At `s_del`, arm pixels alone reach
  62.6 % of prior although arm *state* carries nothing there (Exp 0: 97 %),
  pointing to object→arm rendering leakage (occlusion/shadow) that masking the
  input does not remove. The `s_time` check reuses the `s_del` prediction,
  which cannot hold at `s_time` (the arm is only ε-withdrawn and still reads the
  action).

## 2026-09-29 — Experiment E: 4 DECOY samples unmatched

`paired_gap` dropped 4 DECOY samples with no INTERACT partner (cylinder
validator exclusions differ per condition, as the inclusion policy allows); G
uses 611 matched pairs. Reported, not a deviation of method.

## 2026-10-01 — Experiment H: paired between-model effect reported alongside the registered unpaired one (addition, before H ran)

- **What:** `exp_h.py` reports C3(b) as unpaired Cohen's d between the
  physics-corrupted and base models' per-rollout scores. Every ladder model is
  generated from the *same* held-out (action, seed) tuples (§9-H), so a paired
  estimator -- per-tuple difference, d = mean/sd, seeded percentile-bootstrap
  CI -- is the natural one and matches the paper's paired G.
- **Change:** `paired_effect()` added; written as
  `between_model.corrupted_vs_base_paired` next to the unchanged registered
  entry, printed side by side by `exp_h.py` and `analyze.py --exp h`.
- **Status:** made and committed **before** any Experiment H generation or
  score existed (H launched the same day; see the job chain in HANDOFF §0).
  The registered unpaired d remains the pre-registered quantity; the paired d
  is an addition, not a replacement. **Confirmatory impact: none** (C1/C2 are
  Experiment E).

## 2026-10-02 — Experiment H: per-item generation validator used the MIN consecutive-frame difference (bug fix, after seeing data)

- **What:** `wm_generate.validate_item` required every consecutive pair of
  generated frames to differ by > 1.0 grey level on average. The first two
  frames of every WM clip sample the 0.1 s hold phase before the arm moves, so
  the REAL video is still there, and a correct generation (difference 0.007,
  matching the real clip) failed. All 75 generation tasks aborted on their
  first five items (§13.5 fail-fast) with the trained models otherwise fine
  (S1 determinism / two-action divergence / temporal-motion checks all
  passed; |gen − real| ≈ 0.16 grey levels on frame 0).
- **Fix:** the check now uses the clip-MEAN consecutive difference, the same
  statistic as the S1 `temporal_motion` check (a "model emits a still"
  detector); the max is reported alongside. Existing items are re-validated
  with the current checks on a re-run instead of being skipped as done.
- **Confirmatory impact: none.** This is a pipeline-sanity validator, not a
  registered metric; no H score existed when it was changed.
- **Addendum (same day, before any H score):** (a) `unity/validate.py` carried
  an identical min-based copy of the check; both now use the clip mean with
  **one** floor, 0.5, the S1 `temporal_motion` threshold (the per-item copy had
  used 1.0 on the same statistic, so a correct free-space control could have
  failed one and passed the other); a max-difference check is reported
  alongside. (b) `exp_h.py` previously scored every item on disk, including
  those that failed per-item validation (the generator saves before it
  validates); it now reads the shard manifests and drops failures, and refuses
  to score a model whose generation did not finish. (c) `exp_h.py` now refuses
  non-finite scores (which `cohens_d` would have reported as d = 0) and refuses
  a ladder whose models were scored on different tuple sets. (d) Registered
  `n_rollouts_per_model: 1500` and `controls.n_rollouts: 300` are not
  reachable: the held-out test split of the box corpus holds **193** tuples, so
  every ladder model and every control is scored on the same 193.

## 2026-10-02 — Reporting notes (no change made)

- `analyze.py`'s C1/C2 verdicts use the two-sided 95 % percentile CI
  (97.5th percentile as the upper bound) where prereg §3 says "one-sided 95 %
  CI". This is the stricter reading; the Holm p-values use the one-sided
  inversion. Neither was borderline (C1 upper bound 0.188 vs 0.25; C2 upper
  0.115 vs 0.181). Left as is.
- `results/exp_e/results.json` `G.pooled.n_pairs` is the seed-pooled count
  (5 seeds × 611 = 3055); unique held-out pairs per seed = 611.
- Architecture A resume (`idm.train_idm`) does not reproduce the uninterrupted
  shuffle order bitwise after a relaunch (persistent-worker iterator seed);
  no production Architecture A item was ever resumed (0 restarts in the 30
  logs), so every reported IDM is an uninterrupted run. `resume.pt` now carries
  a config fingerprint and refuses a mismatched relaunch.

## 2026-10-07 — Experiment K: the contact-free arm reference still had contact (bug fix, before any K data)

- **What:** the first compliant-arm calibration (`calibrate_compliance.py`,
  job 65363481) reported 0.00 mm arm deflection at every gain, including gains
  too soft to track the path at all. `scene._simulate(disable_arm_object_contact=True)`
  zeroed only the paddle's `contype`. MuJoCo also makes a contact when the
  object's `contype` (4) matches the paddle's `conaffinity` (4), so the
  "contact-free" reference still had contact and was identical to the real
  rollout.
- **Fix:** zero (and restore) both bitmasks. The paddle has no other contacts.
- **Impact:** position-control mode only. The main study, H and H2 use the
  kinematically clamped arm, which never builds this reference. The K.1 rule
  is unchanged; the calibration is re-run with the fixed instrument before any
  K corpus, IDM or statistic exists. The failed calibration's record is kept
  in the job log.
