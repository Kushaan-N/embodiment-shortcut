"""Experiment A -- contact-regime delta distribution (§9-A).  CPU, minutes.

Measures how far the object actually moves between consecutive *video* frames
while it is being pushed, and turns that into the two numbers the rest of the
project is calibrated against:

  delta_pos_min, delta_rot_min : 5th percentile of the nonzero active-phase
      deltas, per geometry and pooled.  Experiment B sweeps from 0.1x to 10x
      these, and Gate B asks whether the encoder can resolve them.

  settle_lin_speed, settle_ang_speed : the settling cutoff of §6.4, derived as
      "the speed at which the object moves less than delta_min per video
      frame".  Never hardcoded -- §9-A is explicit that 1 mm / 1 degree are
      not acceptable defaults.

Deltas are measured at ``config.FRAME_STRIDE``, not at the physics timestep.
See the FRAME_STRIDE comment in config.py: at 500 Hz the 5th-percentile delta
is a fraction of a micron, and Gate B would then be failing on the integrator
step size rather than on anything about DINO.

    python exp_a_deltas.py [--n 100] [--out results/exp_a]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config as C
import distances as D
import provenance
import scene

EXPERIMENT = "exp_a_deltas"


def active_phase_mask(obj_vel: np.ndarray, contact_start: int,
                      peak_fraction: float = 0.01) -> np.ndarray:
    """Steps belonging to the contact-driven active phase.

    Self-normalising: a step is active if the object's linear speed exceeds
    ``peak_fraction`` of this rollout's own peak.  Defining the active phase
    with an absolute speed cutoff would be circular, since the absolute cutoff
    is what this experiment exists to produce.
    """
    lin = np.linalg.norm(obj_vel[:, :3], axis=1)
    peak = float(lin.max())
    if peak <= 0:
        return np.zeros(lin.size, dtype=bool)
    mask = lin > peak_fraction * peak
    mask[: max(contact_start, 0)] = False
    return mask


def rollout_deltas(r: dict, stride: int):
    """Per-video-frame deltas over the active phase.

    Returns ``(dpos, drot_quotiented, drot_raw)``.  Both rotation variants are
    kept because they answer different questions and §7.2 requires reporting
    both:

    * quotiented -- the *identifiability* scale (Experiments B, C, D).  It is
      identically zero for the sphere by construction.
    * raw geodesic -- the *physical motion* scale, which is what the settling
      criterion needs: a rolling sphere is plainly still moving even though its
      settled pose is rotation-invariant.
    """
    geometry = r["geometry"]
    size = C.GEOM_SIZE[geometry]
    poses = r["obj_poses"]
    forces = r["contact_forces"]
    hit = forces > C.TOLERANCES.contact_force_zero_n
    contact_start = int(np.argmax(hit)) if hit.any() else 0
    mask = active_phase_mask(r["obj_vel"], contact_start)

    dpos, drot_q, drot_raw = [], [], []
    for t in np.nonzero(mask)[0]:
        u = t + stride
        if u >= poses.shape[0] or not mask[u - 1]:
            continue
        dp, dr = D.pose_distance(poses[t], poses[u], geometry, size, quotient=True)
        _, dr_raw = D.pose_distance(poses[t], poses[u], geometry, size, quotient=False)
        dpos.append(dp)
        drot_q.append(dr)
        drot_raw.append(dr_raw)
    return np.asarray(dpos), np.asarray(drot_q), np.asarray(drot_raw)


def _pct_nonzero(x: np.ndarray, q: float = 5.0, eps: float = 0.0) -> float:
    nz = x[x > eps]
    if nz.size == 0:
        raise RuntimeError("no nonzero deltas -- the active phase detector found nothing")
    return float(np.percentile(nz, q))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="contact-heavy rollouts per geometry")
    ap.add_argument("--out", type=Path, default=C.RESULTS_ROOT / "exp_a")
    ap.add_argument("--seed", type=int, default=C.MASTER_SEED)
    args = ap.parse_args()

    stride = C.FRAME_STRIDE
    rng = np.random.default_rng(args.seed)
    arrays: dict = {}
    per_geom: dict = {}
    print(f"Experiment A: {args.n} rollouts/geometry, deltas at stride={stride} "
          f"({C.VIDEO_FPS:g} fps)")

    for geometry in C.GEOMETRIES:
        dpos_all, drotq_all, drotr_all, peak_lin, peak_ang = [], [], [], [], []
        n_contact = 0
        for i in range(args.n):
            # Contact-heavy: the whole action range contacts by construction
            # (T10 verifies it per rollout), so the action distribution is not
            # biased here -- biasing it would make delta_min unrepresentative.
            action = scene.sample_action(rng)
            r = scene.rollout(action, seed=i, condition="INTERACT", geometry=geometry,
                              render=False, thresholds=None)
            if r["contact_end"] <= 0:
                continue
            n_contact += 1
            dp, drq, drr = rollout_deltas(r, stride)
            dpos_all.append(dp)
            drotq_all.append(drq)
            drotr_all.append(drr)
            lin = np.linalg.norm(r["obj_vel"][:, :3], axis=1)
            ang = np.linalg.norm(r["obj_vel"][:, 3:], axis=1)
            peak_lin.append(float(lin.max()))
            peak_ang.append(float(ang.max()))
            if (i + 1) % 25 == 0:
                print(f"  {geometry:9s} {i + 1}/{args.n}")

        dpos = np.concatenate(dpos_all)
        drot_q = np.concatenate(drotq_all)
        drot_r = np.concatenate(drotr_all)
        arrays[f"dpos_{geometry}"] = dpos
        arrays[f"drot_quotiented_{geometry}"] = drot_q
        arrays[f"drot_raw_{geometry}"] = drot_r
        arrays[f"peak_lin_{geometry}"] = np.asarray(peak_lin)
        arrays[f"peak_ang_{geometry}"] = np.asarray(peak_ang)

        d_pos_min = _pct_nonzero(dpos, 5.0)
        d_rot_raw_min = _pct_nonzero(drot_r, 5.0)
        # Quotiented rotation is identically zero for the sphere; there is no
        # 5th percentile of a set with no nonzero elements, and inventing one
        # would be a hardcoded threshold.  Record None and say so.
        quotient_defined = bool((drot_q > 0).any())
        d_rot_min = _pct_nonzero(drot_q, 5.0) if quotient_defined else None

        per_geom[geometry] = {
            "n_rollouts": n_contact,
            "n_deltas": int(dpos.size),
            "delta_pos_min": d_pos_min,
            "delta_rot_min": d_rot_min,
            "delta_rot_min_quotient_defined": quotient_defined,
            "delta_rot_raw_min": d_rot_raw_min,
            "delta_pos_median": float(np.median(dpos)),
            "delta_rot_raw_median": float(np.median(drot_r)),
            "delta_pos_p95": float(np.percentile(dpos, 95)),
            "delta_rot_raw_p95": float(np.percentile(drot_r, 95)),
            "peak_lin_speed_median": float(np.median(peak_lin)),
            "peak_ang_speed_median": float(np.median(peak_ang)),
            # The settling cutoff IS the delta_min scale expressed as a speed:
            # below it, the object moves less than the smallest meaningful
            # amount per video frame.  Angular settling uses the RAW rotation
            # delta, because settling is a question about physical motion, not
            # about pose identifiability.
            "settle_lin_speed": d_pos_min / (stride * C.SIM_DT),
            "settle_ang_speed": d_rot_raw_min / (stride * C.SIM_DT),
        }
        rot_str = "n/a (rotation-invariant)" if d_rot_min is None else \
            f"{np.degrees(d_rot_min):.4f} deg"
        print(f"  {geometry:9s} n={n_contact:4d}  delta_pos_min={d_pos_min * 1e3:.4f} mm  "
              f"delta_rot_min={rot_str}  "
              f"settle_lin={per_geom[geometry]['settle_lin_speed'] * 1e3:.2f} mm/s  "
              f"settle_ang={np.degrees(per_geom[geometry]['settle_ang_speed']):.2f} deg/s")

    pooled_pos = np.concatenate([arrays[f"dpos_{g}"] for g in C.GEOMETRIES])
    # Pooling the sphere's identically-zero quotiented rotation would drag the
    # pooled percentile to zero, so it is excluded and the exclusion recorded.
    rot_geoms = [g for g in C.GEOMETRIES
                 if per_geom[g]["delta_rot_min_quotient_defined"]]
    pooled_rot = np.concatenate([arrays[f"drot_quotiented_{g}"] for g in rot_geoms])

    thresholds = {
        "delta_pos_min": {g: per_geom[g]["delta_pos_min"] for g in C.GEOMETRIES},
        "delta_rot_min": {g: per_geom[g]["delta_rot_min"] for g in C.GEOMETRIES},
        "delta_rot_raw_min": {g: per_geom[g]["delta_rot_raw_min"] for g in C.GEOMETRIES},
        "settle_lin_speed": {g: per_geom[g]["settle_lin_speed"] for g in C.GEOMETRIES},
        "settle_ang_speed": {g: per_geom[g]["settle_ang_speed"] for g in C.GEOMETRIES},
        "pooled_delta_pos_min": _pct_nonzero(pooled_pos, 5.0),
        "pooled_delta_rot_min": _pct_nonzero(pooled_rot, 5.0),
        "rotation_pooling_includes": rot_geoms,
        "frame_stride": stride,
        "video_fps": C.VIDEO_FPS,
        "percentile": 5.0,
        "source_run": EXPERIMENT,
        "per_geometry_detail": per_geom,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    provenance.save_run(args.out, EXPERIMENT, arrays,
                        seeds={"master": args.seed, "n_per_geometry": args.n},
                        extra={"thresholds": thresholds}, arrays_name="delta_dist.npz")
    with open(args.out / "thresholds.json", "w") as fh:
        json.dump(thresholds, fh, indent=2, sort_keys=True)
    print(f"\nwrote {args.out / 'delta_dist.npz'}")
    print(f"wrote {args.out / 'thresholds.json'}  <- every downstream threshold reads this")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
