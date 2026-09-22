# Handoff: running OG-AF on Unity

**State as of 2026-09-21.** Target venue: **ICML 2027** (~late Jan 2027).
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

---

## 1. Day 0 on Unity

```bash
# workspace (NOT $HOME -- see unity/env.sh for why)
export OGAF_DATA=$(ws_allocate ogaf 30)/data
echo "export OGAF_DATA=$OGAF_DATA" >> ~/.bashrc     # or record it somewhere
ws_list -v                                          # note the expiry date

git clone https://github.com/Kushaan-N/embodiment-shortcut
cd embodiment-shortcut

# mujoco has no 3.14 wheels -- 3.11
module load python/3.11                             # or whatever Unity exposes
uv venv --python 3.11 .venv
source unity/env.sh                                 # sets caches OFF $HOME
uv pip install --python .venv/bin/python mujoco numpy scipy scikit-learn \
    statsmodels imageio imageio-ffmpeg tqdm torch torchvision transformers \
    timm pytest pyyaml huggingface_hub
```

**Pre-stage DINOv3 before any job runs.** Every sbatch script sets
`HF_HUB_OFFLINE=1`, so a cache miss is not a slow download — it is a hard
failure at model load, *after* the job has queued and been allocated a GPU.
Compute nodes have no outbound network. Do this on a login node:

```bash
source unity/env.sh                    # so HF_HOME points at the workspace
export HF_TOKEN=<your token>           # the DINOv3 repo is GATED
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
import config as C
p = snapshot_download(C.GATE_B_SELECTED_ENCODER)
print("staged at", p)
PY
```

`/datasets/ai/dinov2` is **not** a substitute. That is DINOv2, which was
measured and rejected at 0.55 sigma against a 3 sigma threshold.

Then verify, on a compute node (several checks are meaningless on a login node,
especially the EGL render):

```bash
srun -p gpu --gres=gpu:1 --time=00:15:00 \
  .venv/bin/python unity/preflight.py --time 08:00:00 --partition gpu
.venv/bin/python -m pytest tests/ -q          # expect 101 passed, ~4 s
```

`preflight.py` checks, among others: the Gate B encoder resolves from cache
(required), caches point off `$HOME` (warn), and a **real one-frame MuJoCo EGL
render** — headless GL silently falls back to osmesa, which works but is ~10x
slower and would corrupt every wall-clock estimate you size later stages from.

---

## 2. Run order

Every command below was checked against the actual argparse surface on
2026-09-21. Costs are from the measured/estimated table in `README.md`.
Each :hand: is a human checkpoint. **If a gate fails, STOP** and report the
measured number against its threshold.

### Human review, before any compute

```bash
.venv/bin/python scripts/contact_sheet.py     # ~10 s
# :hand: INSPECT results/contact_sheet/*.png -- outstanding since build step 2
.venv/bin/python protocol.py
# :hand: 13 fields that were the implementing agent's decision, not a paper's.
#    Load-bearing: clip_length (16, adapted from VPT's 128) and backbone_init
#    (ImageNet-pretrained; a randomly-initialised ResNet-50 would be the
#    strawman T1 forbids).  These back the C1 claim -- review BEFORE the
#    Architecture A trainings.
```

### Corpus — CPU, ~2-3 node-hours

`--clips` is required: A-std is a clip model (MultiWorld §B.2 trains a
*bidirectional* IDM following VPT, so the standard metric's input regime is a
clip, not a frame pair), and A-clip-del is the T8 ablation.

```bash
for g in box sphere cylinder; do
  .venv/bin/python datasets.py --geometry $g --shards $(seq 0 19) --clips
done
.venv/bin/python validate_corpus.py           # T11 + T7    :hand:
```

Storage: frames ~11 GB (memmapped), A-std clips ~8-12 GB compressed.

### IDMs — Architecture B first (cheap, surfaces data bugs), then A

```bash
.venv/bin/python exp_masking.py --train                    # §8.4, 5 seeds
for v in B-std B-del B-time; do for s in $(seq 0 9); do
  .venv/bin/python train_idm.py --variant $v --seed $s
done; done
# Architecture A: 4 variants x 5 seeds = 20 items -> array 0-19
sbatch --array=0-19 -p gpu,gpu-preempt unity/train_idm.sbatch
```

Cost: Arch B probes + masking ~5-15 GPU-h on an A10G; Arch A ~20-60 GPU-h on
A100/L40S.

### Floors, power, and the pre-registration

```bash
.venv/bin/python exp_d_floors.py              # INTERACT held-out only
.venv/bin/python power.py
cp prereg_template.md prereg.md && $EDITOR prereg.md
```

Fill **every** `<<...>>` from the A/C/D/power outputs. Then:

```bash
gh repo edit --visibility public               # REQUIRED -- see below
git add prereg.md && git commit -m "pre-registration (frozen before Experiment E)"
git push                                       # :hand: PUBLIC timestamp
.venv/bin/python prereg_lock.py                # must report PASS
```

**The repo must be public.** `prereg_lock.py` checks that the prereg commit
reached a public remote (§3.7) and there is no skip flag. The reason is not
ceremony: a purely local or private commit is rewritable, so it is weak
evidence that you froze the thresholds before seeing the result.
`analyze.py` keeps Experiments E and H **sealed** until the lock passes.

Optionally also file the same document as an OSF registration — ten minutes for
an external timestamp nobody in this subfield has (§11).

### The decisive experiments — ~5 GPU-h

```bash
.venv/bin/python exp_e_confound.py            # C1, C2, T5 -- the whole paper
.venv/bin/python exp_f_friction.py --generate && .venv/bin/python exp_f_friction.py
.venv/bin/python exp_g_lipschitz.py
.venv/bin/python analyze.py --exp all         # full decision table    :hand:
```

### Experiment H — ~100-200 GPU-h, the budget

Do not start until 0-G pass human review. This is the only stage that needs a
trained world model, and it carries its own claim (C3(b), between-model
ranking).

```bash
# wm_train.py --geometries defaults to ALL THREE, so render all three or it
# starves.  wm_generate.py and wm/exp_h.py default to box.
for g in box sphere cylinder; do
  .venv/bin/python wm/render_clips.py --geometry $g --shards $(seq 0 19)
done
.venv/bin/python wm/vae.py                    # verify_fidelity must pass
bash unity/wm_train.sbatch --smoke            # S1: proves RESUME, not restart
sbatch --array=0-2 -p gpu,gpu-preempt unity/wm_train.sbatch
bash unity/wm_generate.sbatch --s1            # determinism + 2-action divergence
sbatch --array=0-49 -p gpu,gpu-preempt unity/wm_generate.sbatch
.venv/bin/python wm/exp_h.py
```

`wm_train.sbatch` is **built to be killed**: it trains to a wall-clock budget,
checkpoints on USR1, exits 0, and self-requeues to resume. A 150k-step run is
assembled from however many short windows the partition gives you.

The determinism and two-action divergence checks run at **S1, not S2** —
action conditioning being unwired is the one bug that invalidates all
downstream H data, and two generations are enough to detect it.

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

## 5. Related work — one thing to check before writing

Novelty was surveyed 2026-09-21 and the core contribution is unpublished: no
paper claims action-following is confounded by embodiment rendering, uses a
DECOY-style control, or proposes a delayed-horizon object-grounded variant.

- **arXiv 2608.24885**, "Do Robotic World Models Really Follow Actions?"
  (Aug 2026) — diagnoses action-following, but its metric is pose-aware NDTW on
  **end-effector trajectories**. It measures arm kinematics deliberately. Cite
  it as the live example of the practice being critiqued.
- **arXiv 2606.15032**, decision-making-centric position paper — argues
  "pipeline entanglement": an IDM-based action-recovery claim is a claim about
  the whole pipeline, not the world model. Argues abstractly what this project
  measures concretely.
- **arXiv 2604.01985**, World Action Verifier — **READ IN FULL before writing
  related work.** It uses a sparse IDM with feature masking and its
  Proposition 3.1 reportedly argues this keeps verification on "genuine action
  imprints rather than pose artifacts." It points the other way (restricting to
  *agent-centric* features rather than removing the arm) and is a method paper,
  not a measurement critique — but if it already names the pose shortcut, the
  framing shifts from "we identified this" to "we measured and corrected it."

Adjacent papers appeared in 2602, 2603, 2604, 2605, 2606, 2607 and 2608 —
roughly one a month. Novelty is intact; it will not stay that way indefinitely.

## 6. Still open

- `weights_manifest.json` does not exist yet; `preflight.py` reports its
  absence as a warning. Generate it once on a machine where you trust the
  download (recipe in `unity/README.md`) and point it at the **DINOv3** cache.
- `unity/README.md`'s cluster facts were verified 2026-08-17. **Re-verify at
  S0** — a stale policy assumption costs an allocation, not a warning.
- Reciprocal reviewing at ICML/ICLR needs a previously-published co-author,
  and is on the critical path (§2).
- `README.md`'s status table still lists stage 7 as "not run". Fix it when you
  re-run `validate_corpus.py`.
