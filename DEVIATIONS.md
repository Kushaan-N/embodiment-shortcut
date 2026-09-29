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
