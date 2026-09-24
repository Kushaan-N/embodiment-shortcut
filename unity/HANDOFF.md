# Handoff: running OG-AF on Unity

**State as of 2026-09-24.** Target venue: **ICML 2027** (~late Jan 2027).
Workshop route abandoned — this is one run, one pre-registration, straight to
the full paper including Experiment H.

Read `unity/README.md` first for cluster mechanics (the 1-hour default time
limit, the 2-hour `gpu-preempt` kill, workspace expiry, the S0-S4 escalation
ladder). This file is project state and run order; that file is the cluster.

---

## 0. Where the project actually is

Done and passing. Measured values live in `results/*/results.json` and are
committed; `python analyze.py --exp all` prints every gate against its
threshold.

| Stage | Gate | Result |
|---|---|---|
| `distances.py` + tests | 101 unit tests | PASS |
| `scene.py` + contact sheet | visual inspection | **NOT REVIEWED** |
| Exp 0 — state oracles | Gate C0 | PASS |
| Exp A — contact deltas | thresholds written | done |
| Exp B — encoder floor | Gate B | PASS (DINOv3 only) |
| Exp C — injectivity | Gate C | PASS |
| Corpus + `validate_corpus.py` | T11, T7 | **PASS, but STALE — see below** |
| Everything from Exp D onward | — | not run |

**The headline numbers you already own:**

- **Gate C0** — arm-only recovers the action to **1.0-1.2 % of prior** at
  `s_std`, and does so *identically in INTERACT and DECOY* (difference 0.000).
  At `s_del` the arm carries essentially nothing (97-98 % of prior) while the
  settled object still carries real signal (11-44 %). The shortcut channel
  exists and OG-AF has something to measure. Cost: zero GPU-hours.
- **Gate B** — DINOv3/CLS resolves `delta_pos_min` at **+13.90 sigma**;
  DINOv2-large fails the same gate on byte-identical renders (+0.55). A 25x gap
  between encoders. *Caveat that must travel with every rotation-bearing
  distance:* the rotation channel fails at `delta_rot_min` for **every** encoder
  tested (DINOv3/CLS: -1.50 sigma) and first clears 3 sigma at ~1.11 deg, about
  7.9x `delta_rot_min`. The DINO state distance in Exp H is blind to
  sub-degree reorientation. Recorded in `config.GATE_B_EVIDENCE`.
- **Gate C** — unidentifiable fraction at `s_del`, eps_a = 0.10: box 0.0 %,
  sphere 2.3 %, cylinder 3.4 %, against a pre-declared 25 % threshold. T3 does
  not bite at this horizon.

### Two things that are wrong right now

1. **The corpus gate is stale.** `validate_corpus` PASSED 2026-08-17 23:40.
   The corpus on disk was regenerated 2026-08-18 12:05 — 12.5 hours later,
   during a masking debug. The committed report describes 180 tuples for box;
   the shards on disk hold 24, box only. The T11/T7 PASS does not certify the
   data it sits next to. **Re-run it after the real corpus build** (step 4).
   Protocol requires this gate to pass *before any IDM trains*.

2. **What is on disk is a smoke corpus**, not a corpus: `box` only, one shard,
   24 tuples, 55 MB. Target is 2000 tuples x 3 geometries x 3 conditions.
   Do not train anything on it.

3. **A 2026-09-24 static audit found and fixed defects that would have crashed
   or invalidated every GPU stage** (A-clip-del trained on A-std's clips; frame
   stores never built; ResNet-50 weights never pre-staged; the committed
   thresholds invisible once `unity/env.sh` was sourced; a pair-id collision
   that would have crashed Experiment E after all its GPU work; smoke runs
   poisoning the production checkpoint dirs; Experiment H's model names,
   corpus and scoring horizon). See **§7** for the list and for the decisions
   it leaves to you.

---

## 1. Day 0 on Unity

```bash
# workspace (NOT $HOME -- see unity/env.sh for why)
export OGAF_DATA=$(ws_allocate ogaf 30)/data
ws_list -v                                          # note the expiry date

git clone https://github.com/Kushaan-N/embodiment-shortcut
cd embodiment-shortcut

# mujoco has no 3.14 wheels -- 3.11
module load python/3.11                             # or whatever Unity exposes
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python mujoco numpy scipy scikit-learn \
    statsmodels imageio imageio-ffmpeg tqdm torch torchvision transformers \
    timm pytest pyyaml huggingface_hub

# EVERY shell, login or job: caches off $HOME, the venv on PATH, the committed
# thresholds pinned, the workspace results tree seeded, logs/ created.
# Persist BOTH lines -- a shell without them is how the encoder lands in
# $HOME/.cache and the job then fails offline after its GPU is allocated.
echo "export OGAF_DATA=$OGAF_DATA"    >> ~/.bashrc
echo "source $PWD/unity/env.sh"       >> ~/.bashrc
source unity/env.sh
```

**Pre-stage BOTH pretrained weights before any job runs.** Every sbatch script
sets `HF_HUB_OFFLINE=1` and compute nodes have no outbound network, so a cache
miss is not a slow download -- it is a hard failure at model load, *after* the
job has queued and been allocated a GPU. Do this on a login node with
`unity/env.sh` sourced (so the caches land on the workspace):

```bash
export HF_TOKEN=<your token>           # the DINOv3 repo is GATED
python - <<'PY'
from huggingface_hub import snapshot_download
import config as C
print("DINOv3 staged at", snapshot_download(C.GATE_B_SELECTED_ENCODER))
PY
# Architecture A's ImageNet-pretrained ResNet-50 (protocol backbone_init):
# torchvision downloads it on first use into $TORCH_HOME/hub/checkpoints/.
python -c "from torchvision.models import resnet50, ResNet50_Weights; \
           resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); print('ResNet-50 staged')"
```

`/datasets/ai/dinov2` is **not** a substitute. That is DINOv2, which was
measured and rejected at 0.55 sigma against a 3 sigma threshold.

Then verify on a compute node (several checks are meaningless on a login node,
especially the EGL render), and run the unit tests there too -- nothing
computes on a login node:

```bash
srun -p gpu --gres=gpu:1 --time=00:15:00 --cpus-per-task=4 --mem=16G \
  python unity/preflight.py --time 08:00:00 --partition gpu
srun -p cpu --time=00:10:00 --cpus-per-task=4 --mem=8G \
  python -m pytest tests/ -q                   # expect 114 passed, a few seconds
```

`preflight.py` checks, among others: both pretrained weights resolve from the
workspace caches (required), the committed `thresholds.json` is resolvable
(required -- never re-derive it on Unity), caches point off `$HOME`
(required), the frame stores exist (warning until the corpus is built), and a
**real one-frame MuJoCo EGL render** -- headless GL silently falls back to
osmesa, which works but is ~10x slower and would corrupt every wall-clock
estimate you size later stages from.

---

## 2. Run order

Every command below was checked against the actual argparse surface on
2026-09-24. Costs are from the measured/estimated table in `README.md`.
Each :hand: is a human checkpoint. **If a gate fails, STOP** and report the
measured number against its threshold.

**Nothing computes on a login node.** CPU steps go through `unity/run.sbatch`
(a CPU allocation by default); one-off GPU steps use the same file with the
hardware on the command line; the big stages have their own array scripts.
`--time` is mandatory on Unity (default 1 h) and every example sets it.
Submit from the repo root with `unity/env.sh` sourced (it creates `logs/`,
which `#SBATCH --output` needs at submit time).

```bash
GPU='-p gpu --gres=gpu:1 --constraint=[a100|l40s]'     # one-off GPU steps
```

### Human review, before any compute

```bash
python scripts/contact_sheet.py               # ~10 s (fine on a login node)
# :hand: INSPECT $OGAF_RESULTS/contact_sheet/*.png -- outstanding since build step 2
python protocol.py
# :hand: 13 fields that were the implementing agent's decision, not a paper's.
#    Load-bearing: clip_length (16, adapted from VPT's 128) and backbone_init
#    (ImageNet-pretrained; a randomly-initialised ResNet-50 would be the
#    strawman T1 forbids).  These back the C1 claim -- review BEFORE the
#    Architecture A trainings.
```

### Verification-subset audit — CPU, minutes, zero GPU

Reruns nothing beyond Experiment 0; recomputes from the arrays it writes.
`results/exp_0/oracle_errors.npz` is **gitignored**, so Experiment 0 runs on
Unity first (~25 min CPU). Note: the committed `results/exp_0/results.json`
was produced with **3** probe seeds; the command below (5) will not reproduce
it byte-for-byte -- decide which record the paper cites (§7).

```bash
sbatch --time=01:00:00 unity/run.sbatch python exp_0_oracles.py --n 700 --seeds 5
sbatch --time=00:20:00 unity/run.sbatch python verification_subset.py
python analyze.py --exp subset
```

See §5 for why this exists and what it buys the paper.

### Corpus — 60 array items on cheap GPUs (EGL), then two CPU jobs

`--clips` is required and renders **both** clip variants: A-std is a clip
model (MultiWorld §B.2 trains a *bidirectional* IDM following VPT, so the
standard metric's input regime is a clip, not a frame pair), and A-clip-del is
the T8 ablation -- which until 2026-09-24 was silently trained on A-std's clips
(§7).

```bash
bash unity/corpus.sbatch --smoke                 # S1, inside salloc (see the file header)
sbatch --array=0-4  unity/corpus.sbatch          # S2: record the wall clock
sbatch --array=0-59 unity/corpus.sbatch          # S4: 3 geometries x 20 shards
# after the array finishes (--dependency=afterok:<jobid>):
sbatch --time=01:00:00 unity/run.sbatch python idm_data.py         # frame stores (CPU)
sbatch --time=01:00:00 unity/run.sbatch python validate_corpus.py  # T11 + T7 + completeness  :hand:
```

`validate_corpus.py` refuses a corpus that is not the pre-registered one: every
(geometry, condition) cell present with tuples 0..1999, every shard's validator
record present, both clip variants present for every tuple, no geometry skipped
by T11, no validator failing on more than 5 %. The frame stores
(`python idm_data.py`) are the step the old run order was missing; every
trainer opens them and died without them.

Storage: frames ~11 GB (memmapped), clips ~2 x 8-12 GB compressed.

### IDMs — Architecture B first (cheap, surfaces data bugs), then A

```bash
sbatch unity/probes.sbatch                    # 30 B-* probes + 45 masking cells, one GPU, sequential
# Architecture A: items 0-19 = 4 variants x 5 seeds; items 20-29 = the
# AUGMENTED A-std/A-del seeds Experiment H scores with (wm/ladder.yaml).
bash unity/train_idm.sbatch --smoke                   # S1, inside salloc
sbatch --array=0-4  -p gpu unity/train_idm.sbatch     # S2: record Elapsed with sacct
sbatch --array=0-29 -p gpu unity/train_idm.sbatch     # full
```

`train_idm.py` writes a per-epoch resume state (`<tag>/resume.pt`, full
model/optimizer/schedule/RNG state, atomic) and continues from it on the next
launch, so Architecture A can be assembled from 2 h `gpu-preempt` windows like
the world models: `-p gpu,gpu-preempt --time=02:00:00` once S2 has measured the
epoch time. Verify the resume at S3 by actually being preempted (the log says
`resumed from ... epoch k/N`). The restart cap (12) only catches an item that
never completes an epoch.

Cost: Arch B probes + masking ~5-15 GPU-h; Arch A (30 items) ~30-90 GPU-h on
A100/L40S.

### Floors, power, and the pre-registration

```bash
sbatch --time=00:30:00 unity/run.sbatch python exp_d_floors.py      # INTERACT held-out only
sbatch --time=00:30:00 unity/run.sbatch python power.py
cp prereg_template.md prereg.md && $EDITOR prereg.md
```

Fill **every** `<<...>>` from the A/C/D/power outputs (`floors.json` and
`power.json` are versioned, so the registration's sources are in git). Then:

```bash
gh repo edit --visibility public               # REQUIRED -- see below
cp -n $OGAF_RESULTS/exp_d/floors.json results/exp_d/ ; cp -n $OGAF_RESULTS/power/power.json results/power/
git add prereg.md results/exp_d/floors.json results/power/power.json
git commit -m "pre-registration (frozen before Experiment E)"
git push                                       # :hand: PUBLIC timestamp
python prereg_lock.py                          # must report PASS (login node: needs network)
```

**The repo must be public, and the lock enforces it.** `prereg_lock.py` passes
only if the prereg commit is an ancestor of the pushed upstream AND that
upstream answers an anonymous HTTPS `ls-remote` (a private remote cannot pass
through a cached login), and only if `prereg.md` contains no `<<...>>`
placeholder. The reason is not ceremony: a purely local or private commit is
rewritable, so it is weak evidence that you froze the thresholds before seeing
the result. `analyze.py` keeps Experiments E and H **sealed** until the lock
passes, and reads `tau_1`/`eps_a` from `prereg.md` (not from `config.py`
defaults, which the lock does not hash).

Optionally also file the same document as an OSF registration — ten minutes for
an external timestamp nobody in this subfield has (§11).

### The decisive experiments — ~5 GPU-h

```bash
sbatch $GPU --time=03:00:00 unity/run.sbatch python exp_e_confound.py   # C1, C2, T5 -- the whole paper
sbatch $GPU --time=03:00:00 unity/run.sbatch python exp_f_friction.py --generate --clips
sbatch $GPU --time=02:00:00 unity/run.sbatch python exp_f_friction.py
sbatch $GPU --time=02:00:00 unity/run.sbatch python exp_g_lipschitz.py
python analyze.py --exp all                   # full decision table              :hand:
```

`exp_f --generate` needs `--clips`: A-std is a clip model and the swept clips
live under their own `_fm<x>` path (they used to share the baseline path, which
made the dose-response flat by construction).

### Experiment H — ~100-200 GPU-h, the budget

Do not start until 0-G pass human review. This is the only stage that needs a
trained world model, and it carries its own claim (C3(b), between-model
ranking).

```bash
# 1. Data.  WM-base / WM-data-poor train on the INTERACT corpus.  The
#    physics-corrupted model needs a friction x3 INTERACT corpus of the SAME
#    size (a 400-tuple sweep would confound it with data-poor).  ladder.yaml
#    says friction_mult 3.0, which is NOT in config.FRICTION_MULTIPLIERS, so
#    exp_f never builds it -- build it explicitly:
OGAF_CORPUS_ARGS="--friction-mult 3.0 --conditions INTERACT" sbatch --array=0-59 unity/corpus.sbatch
# 2. Video-rate WM clips for both corpora (EGL -> any cheap GPU):
CHEAP='-p gpu --gres=gpu:1 --constraint=[2080_ti|1080_ti|titan_x|m40|v100]'
for g in box sphere cylinder; do
  sbatch $CHEAP --time=02:00:00 unity/run.sbatch python wm/render_clips.py --geometry $g --shards $(seq 0 19)
  sbatch $CHEAP --time=02:00:00 unity/run.sbatch python wm/render_clips.py --geometry $g --shards $(seq 0 19) --friction-mult 3.0
done   # the x3 clips land in wm_clips_fm3/, where wm_train.py looks for them
# 3. VAE, then the three training runs (built to be killed; resumes across windows):
sbatch $GPU --time=04:00:00 unity/run.sbatch python wm/vae.py     # verify_fidelity must pass
bash unity/wm_train.sbatch --smoke            # S1 (salloc): proves RESUME, in an isolated dir
sbatch --array=0-2 -p gpu,gpu-preempt unity/wm_train.sbatch
# 4. Generation: one submission per ladder model (WM-base-{100,30,10} are
#    checkpoints of the WM-base run at ladder.yaml's checkpoint_at steps):
OGAF_WM_MODEL=WM-base-10 bash unity/wm_generate.sbatch --s1     # determinism + 2-action divergence
for m in WM-base-100 WM-base-30 WM-base-10 WM-data-poor WM-physics-corrupted; do
  OGAF_WM_MODEL=$m sbatch --array=0-9 -p gpu,gpu-preempt unity/wm_generate.sbatch
done
# 5. One whole-set validation per model after its array (the per-shard ones
#    inside the array write manifest_validate_shardNNN.json), then score:
for m in WM-base-100 WM-base-30 WM-base-10 WM-data-poor WM-physics-corrupted; do
  sbatch --time=00:30:00 unity/run.sbatch python unity/validate.py \
      --dir $OGAF_DATA/generated/$m/box --expected-frames 16 \
      --manifest $OGAF_DATA/generated/$m/box/manifest.json
done
# 6. ladder.yaml controls: the appearance-gap control generates on ABSENT
#    (free-space) clips -- no object, so contact physics cannot diverge and
#    whatever the metrics read is rendering/domain gap.  300 per model.
for g in box sphere cylinder; do
  sbatch $CHEAP --time=02:00:00 unity/run.sbatch python wm/render_clips.py --geometry $g --shards $(seq 0 19) --condition ABSENT
done
for m in WM-base-100 WM-base-30 WM-base-10 WM-data-poor WM-physics-corrupted; do
  OGAF_WM_MODEL=$m OGAF_WM_CONDITION=ABSENT OGAF_WM_N=300 sbatch --array=0-4 -p gpu,gpu-preempt unity/wm_generate.sbatch
done
sbatch $GPU --time=03:00:00 unity/run.sbatch python wm/exp_h.py
python analyze.py --exp h
```

`wm_train.sbatch` is **built to be killed**: it trains to a wall-clock budget,
checkpoints on USR1, exits 0, and self-requeues to resume. A 120k-step run is
assembled from however many short windows the partition gives you. (It now
waits for the trainer to finish writing its checkpoint before requeueing; the
previous version raced it and could lose up to 2000 steps per kill.)

The determinism and two-action divergence checks run at **S1, not S2** —
action conditioning being unwired is the one bug that invalidates all
downstream H data, and two generations are enough to detect it. Run them on a
real checkpoint (`WM-base-10` = step 12 000 exists about an hour into
training; `OGAF_WM_CKPT=<path>` scores any checkpoint), not on the 400-step
smoke model, whose two-action divergence would be near zero for the wrong
reason.

`exp_h.py` scores the **standard** metric at the generated frame nearest
`s_std` (end of push) and OG-AF at the last frame -- the horizons each IDM was
trained on; it needs the augmented A-std/A-del IDMs (train_idm array items
20-29) and voids C3(b) unless the ladder is verified monotone.

---

## 3. Rules that are not negotiable

- **Never tune a threshold, swap an encoder, resample, or re-seed to make a
  gate pass** (§0.1). If a gate fails, that is the result.
- **Validate content, never exit codes.** The expensive failure is a job that
  exits 0 and writes unusable output.
- **`analyze.py` is the only place that turns arrays into verdicts.** Do not
  compute a verdict anywhere else.
- **G is the mean of per-pair differences**, never a difference of condition
  means. `stats.paired_gap` is the only route to G and it refuses unmatched
  data, so the mistake is not reachable — keep it that way.
- **Null results are results.** If C1 is refuted, current practice is sound on
  this axis and the project stops; the surviving paper is C0 + the masking
  decomposition + the floors. That branch is pre-written in
  `prereg_template.md` §3 and is as publishable internally as the positive one.

## 4. Deviations to state in the paper, not hide

- **The arm has 4 DOF, not the spec's 2-3.** With 3 DOF the arm's visible
  configuration at a single horizon is `(x, y)` at constant push height — two
  numbers, which cannot be injective in a 3-dimensional action, so C0 would be
  refuted by dimension counting rather than by physics. Yaw makes
  `(v, theta, d) -> (x, y, yaw)` analytically invertible, which is exactly the
  situation under study. Recorded as `arm_dof_deviation`.
- **The arm is kinematically clamped** (`ARM_CONTROL=kinematic`), the T9
  escalation the spec permits. `arm_qpos[t] == commanded[t]` exactly, so the
  cross-condition arm deviation is 0.0 *by construction, not by tuning* — and
  Exp 0's residual probe confirms that channel carries no action information
  (97 % of prior). The quasi-infinite arm impedance is a modelling trade-off to
  state. `OGAF_ARM_CONTROL=position` switches to high-gain PD and the validator
  then measures the real tracking residual against a contact-free reference.
- **Exp A measures deltas at the video frame rate, not the physics timestep.**
  At 500 Hz the 5th-percentile active-phase delta is sub-micron and Gate B
  would fail as an artefact of the integrator step. `VIDEO_FPS = 20` puts
  `delta_pos_min` at 4.5-5.4 mm, roughly 1-2 px.
- **The DINO rotation blindness above.** State it wherever a rotation-bearing
  distance appears.

## 5. Related work, and the verification-subset result

Surveyed 2026-09-21, re-read from source 2026-09-22. **The core contribution is
unpublished**: no paper claims action-following is confounded by embodiment
rendering, uses a DECOY-style control, or proposes a delayed-horizon
object-grounded variant.

**arXiv 2608.24885**, "Do Robotic World Models Really Follow Actions?"
(Aug 2026) — diagnoses action-following, but its metric is pose-aware NDTW on
**end-effector trajectories**. It measures arm kinematics deliberately. Cite it
as the live example of the practice being critiqued.

**arXiv 2606.15032**, decision-making-centric position paper — argues "pipeline
entanglement": an IDM-based action-recovery claim is a claim about the whole
pipeline, not the world model. Argues abstractly what this project measures.

**arXiv 2604.01985**, World Action Verifier — **read in full 2026-09-22. It is
not prior art, it is the best available foil.** An earlier automated summary of
this paper claimed it "prevents the IDM from exploiting robot pose as a
shortcut" and spoke of "pose artifacts". Those phrases are not in the paper:
`shortcut`, `artifact`, `confound`, `spurious`, `embodiment` and `object
physics` each occur **zero** times. What the paper actually does is assume an
"identifiable verification subset" S and prove (Prop. 3.1) that a sparse inverse
model transfers out-of-support provided

  (i) `z_S^{t+1}` depends only on `(z_S^t, a^t)` **and not on the rest of the
      scene**; (ii) the subset stays on-support when the full transition is not;
  (iii) the action is identifiable from `(z_S^t, z_S^{t+1})`.

Their own reading of S is "the agent's own motion pattern (e.g. joint-angle
trajectories)" — the manipulator channel. So a 2026 paper states the embodiment
shortcut as a **theorem** and builds a method on it, treating it as a robustness
guarantee. For verification it is one. For evaluation, condition (i) *is*
object-blindness.

`verification_subset.py` turns that into a measurement, from Experiment 0's
existing arrays at zero cost. Measured:

| horizon | (i) scene-independent | (iii) subset recovers action | object-only |
|---|---|---|---|
| `s_0` | yes | 97-98 % of prior (fails) | 99-102 % |
| `s_std` | yes | **1.0-1.2 % (holds)** | 11-46 % |
| `s_del` | yes | 97-98 % (fails) | 11-45 % |
| `s_time` | yes | **1.0-1.2 % (holds)** | 11-45 % |

Both conditions hold at **`s_std`** — the horizon the standard metric uses — and
fail at **`s_del`**, the horizon OG-AF proposes. The guarantee and the confound
are the same property read for different purposes, and the two horizons are
formally complementary. Condition (ii) is **not measurable** in this corpus (it
concerns out-of-support transitions the corpus does not contain) and is reported
as unmeasured rather than estimated.

This is descriptive support, never the basis of a confirmatory claim — same
standing as the masking decomposition (§6 of the prereg).

Adjacent papers appeared in 2602, 2603, 2604, 2605, 2606, 2607 and 2608 —
roughly one a month. Novelty is intact; it will not stay that way indefinitely.

## 6. Still open

- `weights_manifest.json` does not exist yet; `preflight.py` reports its
  absence as a warning. Generate it once on a machine where you trust the
  download (recipe in `unity/README.md`) and point it at the **DINOv3** cache.
- `unity/README.md`'s cluster facts were re-verified 2026-09-24 against live
  `sinfo`/`sacctmgr` and the Unity docs. **Re-verify at S0** — a stale policy
  assumption costs an allocation, not a warning.
- Reciprocal reviewing at ICML/ICLR needs a previously-published co-author,
  and is on the critical path (§2).
- **Encoder precision is deliberately left at fp32.** `exp_b_resolution.FrozenEncoder`
  runs the ViT in fp32 at batch 32. bf16 autocast would be roughly 2-3x faster on
  an A100, but Gate B's measured sigmas were obtained in fp32 and the embeddings
  feeding every Architecture B probe would no longer be the ones the gate
  validated. If you want the speed, treat it as a protocol change: re-run
  Experiment B first and record the new sigmas. Do not flip it quietly.
- The one optimisation taken (2026-09-22) is in `embed_stores`: `np.ix_` reads
  only the two horizons needed instead of materialising all four, measured 2.7x
  faster on a 1.1 GB store against an ~11 GB corpus store, with bitwise
  identical output. `tests/test_frame_selection.py` pins it. The ViT forward
  still dominates, so expect a few per cent end-to-end, not 2.7x.

## 7. What the 2026-09-24 audit changed, and what it leaves to you

A static adversarial review of the whole tree before any GPU time (nothing was
executed on the login node; every finding was verified by reading the code).
All fixes are committed file by file with their reasons in the messages. The
ones that would have cost a GPU allocation or produced a wrong result:

- **A-clip-del was A-std.** `datasets.py` rendered one clip (`A-std`) and
  `ClipDataset` ignored its variant, so the T8 ablation would have trained on
  the standard window and "passed" trivially; its index set was also 15 frames
  whenever `s_del == arm_rest` (a collate crash). Every `config.CLIP_VARIANTS`
  clip is now rendered and stored under
  `clips/<variant>/<geometry>/<condition>[_fmX]/`, and `clip_indices` is
  fixed-length by construction (tests pin it).
- **Frame stores were never built** in the run order (`FrameStore` raised
  `run materialise_frames first`; the only callers were Modal and exp_f).
  `python idm_data.py` is the step; `preflight.py` warns when they are missing;
  a store built from other shards than the ones on disk is rebuilt, not reused.
- **ResNet-50 weights were downloaded at model build** (no network on compute
  nodes); pre-staging is in §1 and `preflight.py` requires the file.
- **The committed thresholds and gate records were invisible on Unity**:
  `env.sh` moved `OGAF_RESULTS` to the empty workspace, so every script raised
  `ThresholdsMissing` and `analyze.py` saw no Exp 0/A/B/C; the error's
  suggested fix would have re-measured the thresholds Gates B/C were certified
  against. `env.sh` pins `OGAF_THRESHOLDS` to the committed file and seeds the
  workspace results tree (no-clobber).
- **Experiment E would have crashed after all its GPU work**: pooled geometries
  repeat `tuple_index`, and `stats.paired_gap` refuses duplicate pair ids. Pair
  ids are `stats.pair_id(geometry, tuple_index)` in exp_d/e/g and in the
  confound arrays; a test pins it.
- **The smoke runs poisoned the production checkpoint dirs**: a 1-epoch
  `A-std__seed0__full` and a 400-step `wm/WM-base/final.pt` would have been
  adopted by skip-if-exists (WM-base would never have trained). Both smokes
  write to isolated `_smoke` / `wm_smoke` dirs.
- **`WM-base-100/-30/-10` could not be generated** (no checkpoint mapping) and
  the physics-corrupted corpus could not be built (friction 3.0 is not in
  `FRICTION_MULTIPLIERS`; `render_clips.py` wrote swept clips onto the base clip
  paths and exited 0 having written nothing). `wm_generate.py` resolves ladder
  names to `ckpt_step<checkpoint_at>.pt`; `render_clips.py` writes
  `wm_clips_fmX/` and fails on missing input; `datasets.py --conditions` builds
  the INTERACT-only x3 corpus.
- **`exp_h.py` scored the standard metric at the wrong horizon** (the settled
  last frame, which A-std never saw). Generated items carry `frame_indices`;
  A-std is scored at the frame nearest `s_std`.
- **Every generated shard was marked FAILED** (`validate.py` expected 1 frame),
  and 50 concurrent validators overwrote one manifest.
- **The sbatch scripts called bare `python`** (venv never activated) and their
  TERM traps could never fire (bash defers traps behind a foreground `srun`);
  `wm_train.sbatch` requeued while the trainer was still writing its checkpoint.
  `env.sh` puts the venv on PATH; the scripts background `srun` and wait for the
  PID to be gone; `--open-mode=append` keeps the preemption evidence.
- **Seeds were not independent replicates**: `train_idm.py` built the model
  before seeding, so all seeds shared one initialisation.
- **Experiment F's dose-response was flat by construction**: the embedding cache
  key and the clip path ignored the friction multiplier.
- **WM training was fp16, not the registered bf16**; latents were cached in
  fp16; a cache built for other geometries/fraction was silently reused; the
  wall-clock budget started only after the cache build; a NaN loss ran on.
- **`validate_corpus.py` passed on a 24-tuple smoke corpus** with two
  geometries skipped -- it now checks completeness, clips and exclusion caps.
- **`analyze.py` never read `tau_1` from `prereg.md`** (line-prefix match
  against a mid-line token) and its Holm correction was dead code;
  `prereg_lock` passed without a push and with unfilled placeholders. Fixed;
  T5 stays descriptive (no registered null), so the Holm family is {C1, C2}.
- Atomic writes for `eval.npz` / `metadata.json` / `checkpoint.pt` / the
  texture PNG / `validators.json` (now written before the shard it describes).
- Corpus, probes and one-off steps have sbatch wrappers (`unity/corpus.sbatch`,
  `unity/probes.sbatch`, `unity/run.sbatch`), so no documented command runs on
  a login node.

**Decisions taken 2026-09-24 (second pass), so they are no longer open:**

1. **The ladder run is 120 000 steps.** `train.steps: 120000`; WM-base-100 is
   the run's final state (`final.pt`), 36 000 / 12 000 are 30 % / 10 % of it.
   Saves 20 % of the H training budget that no ladder model used.
2. **The augmented A-std/A-del IDMs stay** (train_idm items 20-29): the ladder
   registers `augmented: true` for the H metrics, and dropping them would be a
   protocol change made for cost.
3. **The controls are implemented.** `appearance_gap`: `wm_generate.py
   --condition ABSENT` generates on free-space clips (no object, physics
   divergence zero by construction) and `exp_h.py` reports every metric there
   as `domain_gap`; `domain_shift`: the DINO gen-vs-real distance on the
   INTERACT generations next to the clean-vs-augmented IDM delta. C3(b)'s
   effect sizes remain unpaired Cohen's d (left as is; noted).
7. **`train_idm.py` resumes.** Per-epoch `resume.pt` (model, optimizer,
   schedule, scaler, history, best, every RNG incl. the loader's shuffle
   generator), atomic; deleted only after `eval.npz` lands. Architecture A
   can run on `gpu-preempt`. Unvalidated on a GPU: the S1 smoke plus one
   deliberate preemption at S3 are its test.

**Still open:**
4. Gate verdicts for C0/B/C/corpus are computed inside the exp scripts (with
   thresholds hardcoded there) and `analyze.py` echoes the stored booleans --
   "analyze.py is the only place arrays become verdicts" is not yet
   mechanically true for those gates.
5. The lock's "commit precedes artifacts" check is mtime-based (a `cp` or a
   re-clone resets it); the public push timestamp is the real anchor.
6. The committed `results/exp_0/results.json` used 3 probe seeds; §2's command
   uses 5.
7. `preflight.py`'s EGL check times context creation plus one frame against a
   0.5 s heuristic; a cold context on a healthy node can trip it -- re-run
   before believing an osmesa diagnosis.
