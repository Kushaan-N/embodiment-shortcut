# The Embodiment Shortcut in Action-Following Metrics

Video world models are routinely evaluated with an **action-following** metric:
train an inverse dynamics model (IDM), run it on generated video, compare the
inferred action to the action the generation was conditioned on. It is treated
as physics-sensitive and appearance-invariant.

**Hypothesis: it is dominated by embodiment rendering, not object physics.**
When the action is an actuator command, the manipulator's pose in the post-state
is a near-direct readout of that action. An IDM minimising action-prediction
error will learn to read the arm and ignore the object — not as a pathology, but
because that is the *optimal solution to the task as posed*. The metric then
scores how faithfully a world model renders the robot, while being largely blind
to whether contact physics was correct.

Naming: *action-following error* for the existing metric, **OG-AF
(Object-Grounded Action-Following)** for the corrected variant, *the embodiment
shortcut* for the confound. Not "Action-Space Divergence" — that term is
occupied in RL.

---

## Status

| Stage | Gate | Result |
|---|---|---|
| 1. `distances.py` + tests | 114 unit tests | **PASS** |
| 2. `scene.py` + validators + contact sheet | visual inspection | reviewed 2026-09-29 (agent); ✋ human sign-off pending |
| 3. Experiment 0 — state oracles | **Gate C0** | **PASS** |
| 4. Experiment A — contact deltas | thresholds written | **done** |
| 5. Experiment B — encoder floor | **Gate B** | **PASS** (DINOv3; DINOv2 fails) |
| 6. Experiment C — injectivity | **Gate C** | **PASS** |
| 7. Corpus + `validate_corpus.py` | T11, T7, completeness | **PASS** 2026-09-29 — 9×2000 tuples, 36k clips, T11 R²≈0, ≤1.1 % excluded |
| 8. IDMs — Architecture A (30) | content checks | **done** 2026-09-29 (Arch B blocked: DINOv3 needs `HF_TOKEN`) |
| 9. Experiment D — floors, + power | headroom vs prior | **done** — A-std 1.9 %, A-del 44.7 % of prior; n ≤ 102/geometry |
| 10. Pre-registration (`prereg.md`, public push) | §0.5 lock | **PASS** 2026-09-29 (`d2fdc9e`) |
| 11. Experiment E — the confound | **C1, C2** (Holm) | **both SUPPORTED** — G(A-std)/floor 0.151 [0.115, 0.188] < 0.25; err(A-del) 44.7 % < 72 %; T5, T8 hold |
| 11. Experiments F, G, masking (exploratory) | — | done; see `DEVIATIONS.md` for two mis-specified predictions |
| 14. Resolution curve (post-hoc, exploratory) | — | OG-AF detects physics changes that move the object ≥ 2.5 mm (d = +0.64 at 2.5–5 mm, +3.27 at 40–80 mm); the standard metric shows nothing below 80 mm and only +0.15 % of prior beyond it |
| 13. Experiment H2 — rebuilt ladder (`prereg_addendum_H2.md`) | C3(b)-H2 | **SUPPORTED** 2026-10-05 — friction ×0.1 (calibrated), object-region ladder monotone (ρ = 1.00); paired d corrupted−base: OG-AF +1.94 [1.73, 2.22] vs standard +0.46 [0.37, 0.56], Δ +1.48 [1.22, 1.79] |
| 12. Experiment H — world-model ladder | C3(a), C3(b) | **done** 2026-10-02 — C3(a) PASS for all 5 models (standard/OG-AF rank ρ ≤ 0.28 vs 0.9); **ladder not monotone (ρ = 0.70 < 0.9) → C3(b) void, as registered** |

Measured gate values are in `results/*/results.json`; `python analyze.py`
prints them all with thresholds and PASS/FAIL.

### Gate C0 (Experiment 0), measured

| Geometry | arm-only @ s_std | object-only @ s_del | arm-only @ s_del | T9 arm residual |
|---|---|---|---|---|
| box | **1.2 %** of prior | 44.5 % | 97.2 % | 97.2 % |
| sphere | **1.1 %** | 11.2 % | 97.1 % | 97.1 % |
| cylinder | **1.0 %** | 41.5 % | 98.4 % | 98.4 % |

Read: the arm alone recovers the action at the standard horizon to ~1 % of the
prior baseline, and does so **identically in INTERACT and DECOY** (difference
0.000). At the delayed horizon the arm carries essentially nothing (97 % of
prior) while the settled object still carries real signal. The shortcut channel
exists, and OG-AF has something to measure. Established for **zero GPU-hours**.

### Gate B (Experiment B), measured — **PASS**, on one encoder only

| encoder / pooling | translation @ δ_min | rotation @ δ_min | verdict |
|---|---|---|---|
| dinov2-large / CLS | +0.55 σ | −1.10 σ | **fail** |
| dinov2-large / mean-patch | +0.19 σ | −1.08 σ | **fail** |
| **dinov3-vitl16 / CLS** | **+13.90 σ** | −1.50 σ | **PASS** |
| dinov3-vitl16 / mean-patch | +7.00 σ | −1.56 σ | PASS |

Gate B requires ≥3 σ for *at least one* encoder/pooling. It is met, but the
headline is the **25× gap between two encoders evaluated on byte-identical
renders**. DINOv2 cannot see a 1.7 px object displacement; DINOv3 sees it at
13.9 σ, and is still at +2.41 σ for a **0.17 px** displacement — its true floor
is below the swept range. Encoder choice is not a detail for anyone building a
feature-space metric on contact-scale geometry.

**Caveat that must travel with every rotation-bearing distance:** the rotation
channel fails at `δ_rot_min` for *every* encoder tested (DINOv3/CLS: −1.50 σ).
It first clears 3 σ at ~**1.11°**, about 7.9× `δ_rot_min`. So the DINO state
distance in Experiment H, and the box/cylinder probes, are effectively blind to
sub-degree reorientation. Gate B passes on translation; this limit is recorded
in `config.GATE_B_EVIDENCE` and belongs in `prereg.md`.

Selection is recorded as a measurement, not a preference:
`config.GATE_B_SELECTED_ENCODER` with the rejected alternatives and their σ.

### Gate C (Experiment C), measured

Unidentifiable fraction at `s_del`, ε_a = 0.10: box 0.0 %, sphere 2.3 %,
cylinder 3.4 % — against a pre-declared 25 % threshold. T3 does not bite at
this horizon.

---

## Run order

Each ✋ is a human checkpoint. **If a gate fails, STOP** and report the measured
number against its threshold. Never tune a threshold, swap an encoder, resample,
or re-seed to make a gate pass (§0.1).

```bash
# environment (mujoco has no 3.14 wheels); on Unity see unity/HANDOFF.md §1
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python mujoco numpy scipy scikit-learn \
    statsmodels imageio imageio-ffmpeg tqdm torch torchvision transformers \
    timm pytest pyyaml huggingface_hub

.venv/bin/python -m pytest tests/ -q                    #  114 tests, a few seconds
.venv/bin/python scripts/contact_sheet.py               #  ~10 s   ✋ inspect the PNGs
.venv/bin/python exp_a_deltas.py --n 120                #  ~2 min  -> thresholds.json
.venv/bin/python exp_0_oracles.py --n 700 --seeds 5     #  ~25 min GATE C0        ✋
.venv/bin/python exp_b_resolution.py                    #  ~5 min  GATE B (GPU)   ✋
.venv/bin/python exp_c_injectivity.py --pairs 240       #  ~4 min  GATE C         ✋

# corpus: 2000 tuples x 3 geometries x 3 conditions, BOTH clip variants
for g in box sphere cylinder; do
  .venv/bin/python datasets.py --geometry $g --shards $(seq 0 19) --clips
done
.venv/bin/python idm_data.py                            #  frame stores (CPU, once)
.venv/bin/python validate_corpus.py                     #  T11 + T7 + completeness ✋

# IDMs: Architecture B first (cheap, surfaces data bugs), then A
.venv/bin/python exp_masking.py --train                 #  §8.4, 5 seeds
for v in B-std B-del B-time; do for s in $(seq 0 9); do
  .venv/bin/python train_idm.py --variant $v --seed $s
done; done
sbatch --array=0-29 -p gpu unity/train_idm.sbatch       #  Architecture A (20 + 10 augmented)

.venv/bin/python exp_d_floors.py                        #  INTERACT only
.venv/bin/python power.py
cp prereg_template.md prereg.md && $EDITOR prereg.md
git add prereg.md && git commit -m "pre-registration" && git push   #  PUBLIC      ✋
.venv/bin/python prereg_lock.py                         #  must PASS (verifies the public push)

.venv/bin/python exp_e_confound.py                      #  the decisive experiment
.venv/bin/python exp_f_friction.py --generate --clips && .venv/bin/python exp_f_friction.py
.venv/bin/python exp_g_lipschitz.py
.venv/bin/python analyze.py --exp all                   #  full decision table     ✋

# Experiment H — do NOT build until 0–G pass human review; full recipe in
# unity/HANDOFF.md §2 (friction x3 corpus, clips, VAE, 3 trainings, 5 generations)
.venv/bin/python wm/vae.py && .venv/bin/python wm/wm_train.py --model WM-base
OGAF_WM_MODEL=WM-base-10 bash unity/wm_generate.sbatch --s1   #  determinism + divergence
```

On Unity every one of these runs under SLURM (`unity/run.sbatch`,
`unity/corpus.sbatch`, `unity/probes.sbatch`, the array scripts) -- nothing
computes on a login node.  `unity/HANDOFF.md` §2 is the authoritative order.

## Cost and wall-clock

| Stage | Substrate | Measured / estimated |
|---|---|---|
| Exp 0, A, C | CPU | < 1 node-hour total (measured: ~30 min on 1 core) |
| Exp B | A10G | < 1 GPU-hour |
| Corpus, 18 k rollouts + renders | CPU ×8 | ~2–3 node-hours (measured ~0.35 s/rollout) |
| Arch B probes (30) + masking (45) | A10G | ~5–15 GPU-hours |
| Arch A trainings (30: 20 + 10 augmented for Exp H) | A100/L40S | ~30–90 GPU-hours |
| D/E/F/G evaluation | A10G | ~5 GPU-hours |
| **Exp H: WM training + generation** | A100 | **~100–200 GPU-hours** |

Everything before H is cheap; H is the budget. The fail-fast ordering exists so
no H dollar is spent until 0–G survive human review.

Storage: frames ~11 GB (memmapped), clips ~2 × 8–12 GB compressed (A-std and
A-clip-del), generated rollouts ~1 GB/model.

## Layout

```
config.py            all constants; Thresholds loader (raises if Exp A hasn't run)
distances.py         §7.2 symmetry-quotiented distances, §7.3 prior baselines
stats.py             paired estimator + hierarchical bootstrap + Holm
scene.py             MJCF, rollout(), every §6 validator
datasets.py          corpus build, (action,seed)-tuple splits, T7 assertion
idm.py               Architectures A and B (no Modal imports)
idm_data.py          frame stores, pair/clip/embedding datasets, §8.4 masking
train_idm.py         one (variant, seed) -> checkpoint + held-out errors
protocol.py          + protocol/multiworld_b2.yaml -- the §0.6 reproduction record
prereg_lock.py       the §0.5 mechanical lock (no skip flag)
exp_*.py             one file per experiment
analyze.py           the ONLY place that turns arrays into verdicts
wm/                  Experiment H: vae.py, wm_train.py, wm_generate.py, exp_h.py, ladder.yaml
unity/               pipeline.sh (one-command chain), preflight.py, validate.py, env.sh,
                     sync_results.sh, sbatch: corpus/probes/train_idm/wm_train/wm_generate/run
tests/               114 tests: distances, priors, stats, splits, validators, clips
```

## Design decisions a reviewer should check first

**The arm has 4 DOF, not the 2–3 the spec asks for.** With 3 DOF the arm's
visible configuration at a single horizon is `(x, y)` at a constant push height
— two numbers, which cannot be injective in a three-dimensional action. C0 would
then be refuted by dimension counting rather than by physics. The yaw joint
makes `(v, θ, d) → (x, y, yaw)` analytically invertible, which is exactly the
"actuator command is a near-direct readout of manipulator pose" situation under
study. Recorded in `metadata.json` as `arm_dof_deviation`.

**The arm is kinematically clamped** (`ARM_CONTROL=kinematic`), the T9
escalation the spec permits. `arm_qpos[t] == commanded[t]` exactly, so the
cross-condition arm deviation is **0.0 by construction, not by tuning** — and
Experiment 0's residual probe confirms the channel carries no action information
(97 % of prior). The quasi-infinite arm impedance is a modelling trade-off to
state in the paper, not hide. `OGAF_ARM_CONTROL=position` switches to high-gain
PD, and the validator then measures the real tracking residual against a
contact-free reference simulation.

**Experiment A measures deltas at the video frame rate, not the physics
timestep.** At 500 Hz the 5th-percentile active-phase delta is sub-micron and
Gate B would fail as an artefact of the integrator step. The meaningful scale is
displacement between consecutive *video* frames (`VIDEO_FPS = 20`), which puts
`delta_pos_min` at 4.5–5.4 mm ≈ 1–2 px — the regime where the gate is a real
question.

**DECOY placement needs no action-dependent rejection.** The annulus is proven
disjoint from the union of *every* reachable push corridor
(`assert_corridor_decoy_disjoint`, clearance 14 mm), so placement is sampled
from an independent RNG stream with no rejection step that could reintroduce
correlation. That is the structural half of T11; `validate_corpus.py` runs the
empirical ridge-regression half before any IDM trains.

**`split_for_tuple` takes no `condition` argument.** T7 (paired-split leakage)
would bias G *toward* the hypothesis, so the leak is made unrepresentable rather
than merely unlikely.

**MultiWorld §B.2 specifies a *clip*, not a frame pair.** The published IDM is
bidirectional, following VPT (non-causal, 128-frame window). So A-std is a clip
model — which is what makes it the faithful reproduction the C1 claim must hold
for, and what makes A-clip-del a meaningful T8 ablation.

## Open items requiring a human

0. **The 2026-09-24 audit** (`unity/HANDOFF.md` §7): the ladder now trains
   120k steps, the augmented IDMs stay, the H controls are implemented, and
   `train_idm.py` resumes per epoch; still open there: the exp_0 seed-count
   discrepancy and gate verdicts computed outside `analyze.py`.

1. **✋ Contact sheet review** (build-order step 2). `results/contact_sheet/`.
2. **13 protocol fields are the implementing agent's decision, not a paper's.**
   `python protocol.py` prints them with justifications. They are loud, not
   silent — but review them before the Architecture A trainings that back C1.
   The load-bearing ones are `clip_length` (16, adapted from VPT's 128) and
   `backbone_init` (ImageNet-pretrained; a randomly-initialised ResNet-50 would
   be the strawman T1 forbids).
3. **DINOv3 is gated on HuggingFace.** `exp_b_resolution.py` reports it as
   unavailable and runs DINOv2 alone rather than substituting — set `HF_TOKEN`
   if you have access.
4. **NeurIPS 2026 workshop CFPs** and the ICML 2027 deadline need verifying
   against current calls (§2). ICLR 2027's Sept 25 2026 deadline is explicitly
   not the target.
5. **Reciprocal reviewing** at ICLR/ICML 2027 needs a previously-published
   co-author — on the critical path (§2).

## Null results are results

The pre-registered "hypothesis refuted" branches are reachable and written out
in advance (`prereg_template.md` §3). If C1 is refuted, current practice is
sound on this axis and the project stops; the surviving paper is C0 + the
masking decomposition + the floors, as a characterisation of the metric. That
outcome is as publishable internally as the positive one.
