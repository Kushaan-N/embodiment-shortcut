# Unity (UMass) plan

Facts below were verified against Unity docs on 2026-08-17 (§13.2).
**Re-verify at S0** — cluster policy changes and a stale assumption here costs
an allocation, not a warning.

## The traps, in the order they bite

1. **The default time limit is 1 hour on every partition.** `#SBATCH --time=`
   is mandatory. Jobs die at exactly 60 minutes with no useful error. This is
   the single most common way to lose a day here — not the 8-hour cap people
   expect, which belongs to `ood-shared` and interactive `salloc`, not to
   `gpu`/`gpu-preempt`.
2. **`gpu-preempt` jobs can be killed after 2 hours.** No work *item* may
   exceed 2h. Shard until that is true; do not rely on `--requeue` to make a
   long item survive.
3. **Workspaces are not backed up.** `ws_allocate` gives 30 days max with
   limited extensions and 15 TB/user/filesystem (`squota`). **No snapshots, no
   recovery.** Run `ws_list -v` weekly and extend early — an expired workspace
   loses every generation irrecoverably.
4. **`--constraint` selects hardware**, e.g. `--constraint=[a100|l40s]`.
   `gpupod-l40s` and `superpod-a100` are restricted and will silently never
   schedule for a general-access account.
5. **`--qos=long`** is needed beyond ~2 days. General-access `gpu` (153 nodes)
   and `gpu-preempt` (80 nodes) allow up to 14 days.
6. **Check `/datasets/ai/` first.** DINOv2 is already mirrored at
   `/datasets/ai/dinov2`; the nvidia mirror was last pulled 2025-12-06.

## Escalation ladder (§13.3)

The script is **byte-identical at every stage**; only the shard range and the
partition list differ. Never edit an sbatch file between stages — that
invalidates every timing measurement you took to size the next one.

| Stage | Size | Where | Gate |
|---|---|---|---|
| S0 | env check | login node | `preflight.py` passes |
| S1 | 1 item | `salloc` | all validators pass; **determinism AND two-action divergence run HERE** |
| S2 | 5 items | batch, `-p gpu` | validators pass; wall-clock recorded |
| S3 | 50 items | batch, `-p gpu,gpu-preempt` | resume-after-preemption verified **by actually being preempted** |
| S4 | full | batch array | — |

S4 `--time` = **3× the S2 median**. Use `-p gpu` alone through S2 so preemption
does not confound first validation.

S3's gate is deliberately empirical: reading the requeue code and believing it
is not the same as watching a job get killed and come back with its outputs
intact. If you never got preempted at S3, S3 did not happen.

## Governing principle

> The expensive failure is not a crash — it is a job that exits 0 and writes
> unusable output. **Validate content, never exit codes.**

Every sbatch here ends with a content check, and `validate.py` writes a
`manifest.json` with a per-item content checksum so a corrupted volume or an
interrupted rsync becomes detectable later rather than silently poisoning the
analysis.

## Commands

```bash
# S0 — login node
export OGAF_DATA=$(ws_allocate ogaf 30)/data
python unity/preflight.py --time 08:00:00 --partition gpu --projected-gb 400

# S1 — interactive, one item, with the checks that matter
salloc -p gpu --gres=gpu:1 --time=02:00:00 --cpus-per-task=8 --mem=64G
bash unity/train_idm.sbatch --smoke
bash unity/wm_generate.sbatch --s1        # determinism + two-action divergence

# S2 — 5 items, gpu only, record the wall clock
sbatch --array=0-4 -p gpu unity/train_idm.sbatch
sacct -j <jobid> --format=JobID,Elapsed,State,MaxRSS

# S3 — 50 items, invite preemption, then verify resume
sbatch --array=0-49 -p gpu,gpu-preempt unity/train_idm.sbatch

# S4 — full sweep, --time = 3x the S2 median
sbatch --array=0-19 -p gpu,gpu-preempt --time=<3x S2 median> unity/train_idm.sbatch
```

## Monitoring

One progress line per item. Abort a task if its first 5 items all fail.
`manifest.json` records counts by failure reason. **>5 % failures → diagnose
before analysing**, never after.

## `weights_manifest.json`

`preflight.py` checksums encoder weights against this file. Generate it once,
on a machine where you trust the download:

```bash
python - <<'PY' > unity/weights_manifest.json
import hashlib, json, pathlib
root = pathlib.Path("/datasets/ai/dinov2")
out = {}
for p in sorted(root.rglob("*.safetensors")):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    out[str(p)] = h.hexdigest()
print(json.dumps(out, indent=2))
PY
```

A truncated download looks complete. Only a checksum catches it.
