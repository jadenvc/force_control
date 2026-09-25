#!/usr/bin/env python3
"""Replay a folder of --dataset-mode sparse FlipUp episodes, reconstructing
the full dense state/wrench stream offline via PyriteEpisodeRecorder.

Motivation: SparseFlipUpRecorder skips mj_getState/wrench extraction/RGB
specifically because those are what made the live 1kHz control loop lag
(see LATENCY_DIAGNOSTICS.md). Offline replay has no real-time deadline, so
paying that cost here is free -- this script re-drives each episode's
recorded command trajectory through the SAME env and records everything the
full recorder normally would, at whatever rate you ask for (need not match
the live collection rate at all).

Usage:
    python replay_flipup_sparse.py --src ~/data/sparse_run.zarr \
        --dst ~/data/sparse_run_dense.zarr [--dataset-hz 1000] [--verify]

For each source episode:
  1. Reconstruct the FlipUpTeleop env from its recorded metadata_json --
     ``episode_attempt.physical_properties``/``start_sample`` for the exact
     book/start-pose that episode used (no RNG/resampling needed -- the
     ACCEPTED values are already there), and ``command_line`` for every
     controller/contact-physics flag (see ENV_KWARGS_FROM_COMMAND_LINE
     below). Device/haptic-only flags (--stiffness, --damping, --scale,
     --axes, ...) are irrelevant here: replay drives the env directly from
     the recorded target stream, never through the haptic/device layer.
  2. env.configure_episode(properties, book_color_rgba, start_position) --
     deterministic given the same compiled model + settle_s/settle_speed,
     so this reproduces the live run's pre-recording state exactly.
  3. Step the recorded ts_pose_command_0 / target_rotvec / gripper_command
     stream tick-by-tick through env.step(), calling the FULL
     PyriteEpisodeRecorder.record_sample() every tick.
  4. Commit to --dst, carrying over success/termination_reason/final task
     metric from the source episode's attrs.

--verify additionally compares the source's OWN recorded ts_pose_fb_0 (the
live achieved trajectory) against the replay's reconstructed one -- since
MuJoCo is deterministic given identical initial conditions and inputs, a
faithful reconstruction should track it closely; a large deviation means
either the env-reconstruction whitelist below is missing something the live
run actually used, or the source dataset predates some later parameter.

KNOWN LIMITATION: env reconstruction uses an explicit whitelist
(ENV_KWARGS_FROM_COMMAND_LINE), not teleop_flipup.py's own env_kwargs-
building code (embedded in a large stateful main(), not factored out for
reuse). A new physics/controller-relevant CLI flag added to teleop_flipup.py
in the future must also be added here, or replay will silently fall back to
that flag's default instead of the value the live run used. Flags that only
affect haptic feel or device mapping don't need to be listed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flipup_teleop import DEFAULT_JOINT_KD, FlipUpTeleop  # noqa: E402
from flipup.physical_properties import PhysicalProperties  # noqa: E402
from pyrite_recorder import PyriteEpisodeRecorder  # noqa: E402


# command_line key -> FlipUpTeleop constructor kwarg, same name unless noted.
# Only CONTROLLER/CONTACT-PHYSICS flags belong here -- anything that only
# changes haptic feel or device axis mapping (--stiffness, --damping,
# --scale, --axes, --force-gain, ...) never reaches the env at all, so
# omitting them is correct, not an oversight.
ENV_KWARGS_FROM_COMMAND_LINE = {
    "tool_kp": "tool_kp",
    "tool_rot_kp": "tool_rot_kp",
    "tool_rot_kd": "tool_rot_kd",
    "force_clip": "force_clip",
    "surface_force_limit": "surface_force_limit",
    "tool_damping": "tool_damping",
    "standoff": "standoff",
    "tip_softness": "tip_softness",
    "tip_softness_max_solref": "tip_softness_max_solref",
    "tip_softness_max_width": "tip_softness_max_width",
    "table_friction": "table_friction",
    "bookend_solref": "bookend_solref",
    "bookend_solimp": "bookend_solimp",
    "bookend_friction": "bookend_friction",
    "book_fixture_solref": "book_fixture_solref",
    "book_fixture_solimp": "book_fixture_solimp",
    "book_fixture_friction": "book_fixture_friction",
    "tip_friction": "tip_friction",
    "approach_compliance_min_kp_ratio": "approach_compliance_min_kp_ratio",
    "book_normal_force_limit": "book_normal_force_limit",
    "tool_kp_axes": "tool_kp_axes",
    "tool_cartesian_kd": "tool_cartesian_kd",
    "noslip_iterations": "noslip_iterations",
}
# Renamed between the CLI dest name and the constructor kwarg.
RENAMED_ENV_KWARGS = {
    "approach_compliance_distance": "approach_compliance_distance_m",
    "approach_compliance_max_speed": "approach_max_speed_mps",
    "settle": "settle_s",
}


def build_env_kwargs(metadata: dict, offscreen=(640, 480)) -> dict:
    command_line = metadata["command_line"]
    episode_attempt = metadata["episode_attempt"]
    kwargs = {"seed": int(metadata["seed"]), "offscreen": offscreen}

    for src_key, dst_key in {**ENV_KWARGS_FROM_COMMAND_LINE, **RENAMED_ENV_KWARGS}.items():
        if src_key in command_line and command_line[src_key] is not None:
            value = command_line[src_key]
            kwargs[dst_key] = tuple(value) if isinstance(value, list) else value

    arm_damping = command_line.get("arm_damping")
    kwargs["joint_kd"] = (
        None if arm_damping is None else DEFAULT_JOINT_KD * float(arm_damping)
    )

    properties_dict = episode_attempt["physical_properties"]
    kwargs["physical_properties"] = PhysicalProperties(**properties_dict)
    # Must be >= the actual recorded book size (configure_episode raises
    # otherwise) -- the live run's own envelope (base_properties * jitter,
    # computed before per-episode randomization) isn't recorded directly,
    # so just size this to the ACTUAL episode's dimensions with margin.
    dims = np.array(
        [properties_dict["length_m"], properties_dict["width_m"], properties_dict["thickness_m"]],
        dtype=float,
    )
    kwargs["collision_envelope_dimensions"] = dims * 1.05
    return kwargs


def replay_episode(source_episode, dst_recorder: PyriteEpisodeRecorder, *, dataset_hz: float, verify: bool):
    metadata = json.loads(dict(source_episode.attrs)["metadata_json"])
    if metadata.get("task_kind", "flipup") != "flipup":
        raise ValueError(
            f"replay_flipup_sparse.py only supports task_kind='flipup', got "
            f"{metadata.get('task_kind')!r}"
        )
    episode_attempt = metadata["episode_attempt"]
    start_sample = episode_attempt["start_sample"]
    start_position = np.asarray(start_sample["position_world_m"], dtype=float)
    book_color = np.asarray(episode_attempt["book_color_rgba"], dtype=float)

    env_kwargs = build_env_kwargs(metadata)
    target_pose = np.asarray(source_episode["ts_pose_command_0"])
    target_rotvec_all = np.asarray(source_episode["target_rotvec"])
    robot_time_stamps_ms = np.asarray(source_episode["robot_time_stamps_0"])
    n_samples = len(target_pose)
    control_freq_hz = float(metadata["controller"]["control_freq_hz"])

    dst_stride = max(1, int(round(control_freq_hz / dataset_hz)))

    with FlipUpTeleop(**env_kwargs) as env:
        env.configure_episode(env_kwargs["physical_properties"], book_color, start_position)
        # configure_episode()'s internal settle loop only runs for a fixed
        # settle_s (2.5s default) -- live, the operator idles holding at
        # tool_home for however long it takes to press S, often much
        # longer, so the arm keeps converging past that fixed window.
        # Without this, replay's "sample zero" state inherits the FULL
        # un-settled residual (measured: an 8.98mm offset, exactly equal to
        # the episode's own recorded settle_error_m) instead of the
        # closer-to-converged state the live recording actually started
        # from. Holding here for several more seconds (the spring-damper
        # controller converges monotonically) closes that gap; --verify
        # confirms how close.
        for _ in range(int(round(5.0 * control_freq_hz))):
            env.step(start_position, n_substeps=1, target_rotvec=None)

        dst_recorder.start_episode(metadata=metadata)
        step = 0
        replayed_fb = np.zeros((n_samples, 7), dtype=float) if verify else None
        for i in range(n_samples):
            target_pos = target_pose[i, :3]
            rotvec = target_rotvec_all[i]
            target_rotvec = None if np.allclose(rotvec, 0.0) else rotvec
            # How many 1ms physics ticks elapsed since the PREVIOUS recorded
            # sample -- derived from timestamps, not control_batch_size
            # (which only reflects a catch-up batch's size, not a
            # --dataset-hz decimation gap). At i=0 this is 0 (sample zero is
            # the pre-first-tick seed, recorded at timestamp_ms=0 by
            # start_recorded_episode() before any command executes).
            #
            # KNOWN APPROXIMATION when the source was recorded with
            # --dataset-hz < --control-freq (decimated): the intermediate
            # per-tick target values between two recorded samples are lost,
            # so holding target_pos/target_rotvec constant across this gap
            # is the best available reconstruction, not an exact one. This
            # is EXACT (no approximation) when --dataset-hz == --control-freq,
            # i.e. every physics tick was recorded live -- the case this was
            # tested against.
            prev_ms = 0.0 if i == 0 else robot_time_stamps_ms[i - 1]
            n_ticks = max(0, int(round((robot_time_stamps_ms[i] - prev_ms) * control_freq_hz / 1000.0)))
            for _ in range(n_ticks):
                env.step(target_pos, n_substeps=1, target_rotvec=target_rotvec)
                step += 1
                if step % dst_stride == 0:
                    dst_recorder.record_sample(
                        env,
                        timestamp_ms=step * 1000.0 / dataset_hz,
                        target_pos=target_pos,
                        target_rotvec=target_rotvec,
                        device_state={"pos": np.zeros(3), "vel": np.zeros(3)},
                        sent_force=np.zeros(3),
                        image_rgb=None,
                        image_capture_time_s=None,
                        image_id=None,
                    )
            if verify:
                replayed_fb[i] = np.concatenate([env.tool_pos, env.tool_quat])

        source_attrs = dict(source_episode.attrs)
        name = dst_recorder.commit(
            success=bool(source_attrs.get("success", False)),
            termination_reason=str(source_attrs.get("termination_reason", "replayed")),
            final_book_angle_deg=float(env.book_angle_deg()),
            final_task_metric_name=source_attrs.get("final_task_metric_name"),
            final_task_metric_value=source_attrs.get("final_task_metric_value"),
        )

        max_deviation_m = None
        if verify:
            source_fb = np.asarray(source_episode["ts_pose_fb_0"])
            n = min(len(source_fb), len(replayed_fb))
            max_deviation_m = float(
                np.max(np.linalg.norm(source_fb[:n, :3] - replayed_fb[:n, :3], axis=1))
            )
        return name, max_deviation_m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="sparse-mode source Zarr dataset")
    ap.add_argument("--dst", required=True, help="destination Zarr dataset (full/dense schema)")
    ap.add_argument("--dataset-hz", type=float, default=None,
                     help="replay recording rate; defaults to the source's own sample_hz")
    ap.add_argument("--episodes", type=str, default=None,
                     help="comma-separated episode names to replay (default: all)")
    ap.add_argument("--verify", action="store_true",
                     help="compare replayed ts_pose_fb_0 against the source's recorded one")
    ap.add_argument("--dataset-min-samples", type=int, default=1)
    args = ap.parse_args()

    import zarr

    src_root = zarr.open(str(Path(args.src).expanduser()), mode="r")
    if src_root.attrs.get("schema_name") != "pyrite_flipup_sparse":
        print(
            f"[warn] {args.src} schema is {src_root.attrs.get('schema_name')!r}, "
            "not 'pyrite_flipup_sparse' -- proceeding anyway, but this script "
            "expects SparseFlipUpRecorder's field set"
        )
    names = (
        args.episodes.split(",") if args.episodes
        else sorted(
            src_root["data"].group_keys(),
            key=lambda n: int(n.rsplit("_", 1)[-1]),
        )
    )
    dataset_hz = args.dataset_hz or float(src_root.attrs.get("sample_hz", 1000.0))

    dst_recorder = PyriteEpisodeRecorder(
        args.dst, sample_hz=dataset_hz, include_rgb=False,
        min_samples=args.dataset_min_samples,
    )

    for name in names:
        episode = src_root["data"][name]
        print(f"[replay] {name} ({len(episode['ts_pose_command_0'])} sparse samples)...")
        out_name, max_dev = replay_episode(
            episode, dst_recorder, dataset_hz=dataset_hz, verify=args.verify
        )
        if out_name is None:
            print(f"  -> discarded (below --dataset-min-samples)")
            continue
        msg = f"  -> {out_name}"
        if max_dev is not None:
            msg += f"  (max ts_pose_fb_0 deviation vs source: {max_dev * 1000.0:.3f} mm)"
        print(msg)


if __name__ == "__main__":
    main()
