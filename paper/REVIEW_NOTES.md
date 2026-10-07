# Review notes for the workshop draft (2026-10-07)

Two independent read-only reviews of `main.tex`: a claim-by-claim audit against
`results/`, and an adversarial reviewer read (weak accept, 6/10, before revision).

## Fixed in this revision
- Abstract no longer claims every threshold was frozen before data existed. It
  now says what was registered, when (2026-09-29 main, 2026-10-05 world-model
  test), and marks the friction/resolution analyses as exploratory.
- Corrected numbers: "up to 4 cm" -> up to 6 cm; corrupted-corpus loss
  "14-16%" -> 8-18% (12.6% overall); A-time CI lower bound -0.01; 56-57% and
  61-64% on generated video; delta_pos is the 5th-percentile per-frame shift;
  the H2 selection rule's 1.25x factor stated.
- C2 described honestly as near-automatic ("better than chance"); T5/T8 moved
  out of the confirmatory paragraph; C3(a)'s <0.9 criterion marked as not
  registered.
- "Fully accounted for by a no-physics control" softened to a paired-bootstrap
  comparison: standard INTERACT-minus-ABSENT -3.4 pts [-8.3, +1.2]; OG-AF +42.6
  [38.8, 46.1]. CIs added to Fig. 3.
- Resolution curve: discloses that x0.1/x3 corpora are included; reports the
  Exp F-levels-only result; notes the >80 mm bin is all x0.1.
- New "object share" column (0.003 standard vs 0.98 OG-AF) and a plain
  explanation of G and the reference error.
- Framing: flatness of the standard metric is presented as a consequence of
  the position-controlled design; compliant-arm and generality limits added.
- Method figure (real renders of the 3 conditions x 4 horizons).
- Bibliography: all four 2026 entries verified on arXiv; related work expanded
  (shortcut learning, causal confusion, robot world models, physics benchmarks).
- AI-assistance statement expanded.

## Open: need new experiments or decisions (not done)
1. A second corruption type (mass, restitution, or a milder friction model) for
   the world-model test; currently one gross corruption.
2. A compliant/force-limited arm variant to show how far the shortcut shrinks.
3. An arm-masked standard-horizon baseline as a scored metric (only the frozen-
   feature probe number, 48% of chance, exists).
4. Real-robot data / a published world model (external validity).
5. Before submission: anonymous mirror of the repo with identifying strings
   scrubbed (the registration documents name the author and the GitHub
   account); venue style file; author list.
