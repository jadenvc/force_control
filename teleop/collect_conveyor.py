#!/usr/bin/env python3
"""
Haptic data collection for the conveyor pick-place task.

The operator's handle drives the WSG50 tip position; the handle's grip axis (or
its button) drives the finger width; simulated contact force is reflected back
into the handle and the fingertip normal load into the grip axis. Episodes are
written as Pyrite Zarr through the same ``pyrite_recorder`` FlipUp uses.

This is deliberately smaller than ``teleop_flipup.py``. It has the device, the
mapping, the haptics and the recording, and it does not have that script's
composited cv2 viewer, video capture, ``--diagnose`` FFT analysis or 6-DoF wrist
control. What it does have is a ``--dry-run`` mode where the scripted heuristic
plays the operator, so the whole path -- randomized belt speed, grasp, force
reflection, dataset commit, validation -- runs and can be checked without
hardware.

    # No device: the scripted heuristic collects three episodes.
    python collect_conveyor.py --dry-run --episodes 3 \
        --dataset ~/data/conveyor_sim_20hz.zarr

    # With an omega attached.
    python collect_conveyor.py --episodes 20 \
        --dataset ~/data/conveyor_sim_20hz.zarr

Every episode draws a new belt speed, belt/bin offset and cube spawn pose, and
the speed is stored both per sample and in the episode attributes.

Keys, when a device is attached: a long button press ends the current episode.
Ctrl-C discards the episode in progress and leaves the dataset intact.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

TELEOP_DIR = Path(__file__).resolve().parent
if str(TELEOP_DIR) not in sys.path:
    sys.path.insert(0, str(TELEOP_DIR))

from conveyor_teleop import (  # noqa: E402
    DEFAULT_FORCE_CLIP,
    ConveyorTeleop,
    predicted_feel,
)

# Measured in simulation over scripted picks at 0.05/0.15/0.30 m/s: the
# translational contact force this task produces is the cube's 1 N weight while
# carrying, with ~20 N transients on grasp and on hitting the belt or bin. That
# is roughly a tenth of FlipUp's book-levering forces, so a gain sized the way
# FlipUp's was renders this task very lightly. See README_conveyor.md; the grip
# axis is where most of this task's force information is.
DEFAULT_STIFFNESS = 1500.0
DEFAULT_SCALE = 4.0
DEFAULT_AXES = "-x,-y,z"
DEFAULT_FORCE_TAU = 0.002
DEFAULT_FORCE_RATE = 120.0
DEFAULT_DAMPING = 30.0
# Fingertip normal load (N) that maps to the device's full grip force.
DEFAULT_GRIP_FORCE_FULL_SCALE = 60.0
DEFAULT_GRIP_FORCE_MAX = 4.0


def build_pos_map(spec: str) -> np.ndarray:
    """Signed permutation matrix mapping device axes to sim axes.

    Its transpose maps sim forces back onto device axes, which is what keeps
    contact resistance opposing the motion that caused it. Same helper and same
    semantics as ``teleop_flipup.build_pos_map``.
    """
    index = {"x": 0, "y": 1, "z": 2}
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"--axes needs 3 entries, got {spec!r}")
    matrix = np.zeros((3, 3))
    for row, part in enumerate(parts):
        sign = -1.0 if part.startswith("-") else 1.0
        name = part.lstrip("+-").lower()
        if name not in index:
            raise ValueError(f"--axes entry {part!r} must name x, y or z")
        matrix[row, index[name]] = sign
    if abs(abs(np.linalg.det(matrix)) - 1.0) > 1e-9:
        raise ValueError(f"--axes {spec!r} reuses an axis; x, y and z must each appear")
    return matrix


class ForceRenderer:
    """One-pole filter, slew limit and magnitude clamp on the reflected force.

    The order is the one teleop_ball's anti-bounce tuning settled on: filter,
    then slew, then clamp. The slew limit targets the contact onset specifically
    and, unlike raising the filter time constant, does not lag the steady force
    or eat passivity margin.
    """

    def __init__(self, gain, tau, rate_limit, clip, timestep):
        self.gain = float(gain)
        self.tau = float(tau)
        self.rate_limit = float(rate_limit)
        self.clip = float(clip)
        self.timestep = float(timestep)
        self.value = np.zeros(3)

    def __call__(self, sim_force_device_axes):
        target = self.gain * np.asarray(sim_force_device_axes, dtype=float)
        if self.tau > 0.0:
            alpha = self.timestep / (self.tau + self.timestep)
            self.value = self.value + alpha * (target - self.value)
        else:
            self.value = target
        if self.rate_limit > 0.0:
            delta = self.value - getattr(self, "_sent", np.zeros(3))
            max_delta = self.rate_limit * self.timestep
            distance = float(np.linalg.norm(delta))
            if distance > max_delta:
                self.value = getattr(self, "_sent", np.zeros(3)) + delta * (
                    max_delta / distance
                )
        magnitude = float(np.linalg.norm(self.value))
        if magnitude > self.clip:
            self.value = self.value * (self.clip / magnitude)
        self._sent = self.value.copy()
        return self.value.copy()


class ScriptedOperator:
    """Stand-in operator for ``--dry-run``: the package's scripted heuristic."""

    def __init__(self, env):
        from conveyor.heuristic import ConveyorHeuristic

        self.agent = ConveyorHeuristic(env)
        self.env = env

    def reset(self):
        self.agent.reset()

    def __call__(self):
        target_pose, width = self.agent.step()
        return target_pose[:3], width, False

    @property
    def state(self):
        return None


class DeviceOperator:
    """Force Dimension handle: absolute position mapping plus a grip axis."""

    def __init__(self, device, position_map, scale, tool_home, use_button_gripper):
        self.device = device
        self.position_map = position_map
        self.scale = float(scale)
        self.tool_home = np.asarray(tool_home, dtype=float)
        self.use_button_gripper = bool(use_button_gripper)
        self._gripper_closed = False
        self._short_presses = 0
        self._long_presses = 0
        self.state = None

    def reset(self):
        self.device.recenter()
        self._gripper_closed = False

    def __call__(self):
        state = self.device.get_state()
        self.state = state
        offset = self.position_map @ (state["pos"] - state["center"])
        target = self.tool_home + self.scale * offset

        if self.use_button_gripper:
            presses = int(state["short_press_count"])
            if presses != self._short_presses:
                self._gripper_closed = not self._gripper_closed
                self._short_presses = presses
            fraction = 0.0 if self._gripper_closed else 1.0
        else:
            fraction = float(state["gripper"])

        long_presses = int(state["long_press_count"])
        finished = long_presses != self._long_presses
        self._long_presses = long_presses
        return target, fraction, finished


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect haptic demonstrations of the conveyor pick-place task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Pyrite Zarr path to append episodes to. Omit to run without recording.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No device: the scripted heuristic plays the operator.",
    )
    parser.add_argument("--control-freq", type=float, default=1000.0)
    parser.add_argument("--dataset-hz", type=float, default=20.0)
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pace the loop to wall-clock time. Defaults to on with a device, "
        "off in --dry-run.",
    )
    parser.add_argument("--max-episode-seconds", type=float, default=60.0)
    parser.add_argument(
        "--auto-finish",
        action="store_true",
        help="End an episode as soon as the cube is placed, missed or dropped.",
    )
    parser.add_argument(
        "--auto-keep",
        action="store_true",
        help="Commit every episode without asking. Implied by --dry-run.",
    )

    belt = parser.add_argument_group("belt")
    belt.add_argument("--conveyor-speed", type=float, help="Pin the belt speed.")
    belt.add_argument(
        "--conveyor-speed-range", type=float, nargs=2, metavar=("MIN", "MAX")
    )
    belt.add_argument(
        "--randomize-layout", action=argparse.BooleanOptionalAction, default=True
    )
    belt.add_argument(
        "--respawn-object",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return the cube to the belt start after a miss or a drop, so the "
        "operator keeps working within one episode.",
    )

    device = parser.add_argument_group("device and haptics")
    device.add_argument("--axes", default=DEFAULT_AXES)
    device.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    device.add_argument("--home", type=float, nargs=3, metavar=("X", "Y", "Z"))
    device.add_argument("--stiffness", type=float, default=DEFAULT_STIFFNESS)
    device.add_argument("--force-gain", type=float, help="Override stiffness/(kp*scale).")
    device.add_argument(
        "--force-source",
        default="contact",
        choices=("contact", "wrist", "estimated", "none"),
    )
    device.add_argument("--force-clip", type=float, default=DEFAULT_FORCE_CLIP)
    device.add_argument("--force-tau", type=float, default=DEFAULT_FORCE_TAU)
    device.add_argument("--force-rate", type=float, default=DEFAULT_FORCE_RATE)
    device.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    device.add_argument("--max-handle-force", type=float, default=8.0)
    device.add_argument(
        "--grip-force-full-scale", type=float, default=DEFAULT_GRIP_FORCE_FULL_SCALE
    )
    device.add_argument("--grip-force-max", type=float, default=DEFAULT_GRIP_FORCE_MAX)
    device.add_argument(
        "--button-gripper",
        action="store_true",
        help="Latch the gripper on short button presses, for a device without a "
        "grip axis (omega.6).",
    )

    robot = parser.add_argument_group("simulated arm")
    robot.add_argument("--tool-kp", type=float, default=16000.0)
    robot.add_argument("--arm-damping", type=float, default=2.5)

    camera = parser.add_argument_group("camera")
    camera.add_argument("--camera", default="third_person_camera")
    camera.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    camera.add_argument(
        "--rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render and store RGB. Disable for a much faster low-dim-only run.",
    )

    args = parser.parse_args(argv)
    if args.realtime is None:
        args.realtime = not args.dry_run
    if args.dry_run:
        args.auto_keep = True
    if args.control_freq <= 0 or args.dataset_hz <= 0:
        parser.error("--control-freq and --dataset-hz must be positive")
    stride = args.control_freq / args.dataset_hz
    if abs(stride - round(stride)) > 1e-9:
        parser.error("--control-freq must be an integer multiple of --dataset-hz")
    args.record_stride = int(round(stride))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    position_map = build_pos_map(args.axes)

    from conveyor.properties import ValueRange

    speed_range = (
        ValueRange(*args.conveyor_speed_range)
        if args.conveyor_speed_range
        else None
    )

    env_kwargs = dict(
        seed=args.seed,
        tool_kp=args.tool_kp,
        arm_damping=args.arm_damping,
        force_clip=args.force_clip,
        belt_speed_m_per_s=args.conveyor_speed,
        randomize_layout=args.randomize_layout,
        respawn_object=args.respawn_object,
        time_limit_s=args.max_episode_seconds,
    )
    if speed_range is not None:
        env_kwargs["belt_speed_range"] = speed_range

    force_gain = (
        args.force_gain
        if args.force_gain is not None
        else args.stiffness / (args.tool_kp * args.scale)
    )
    feel = predicted_feel(
        args.tool_kp, args.scale, force_gain, args.damping, args.control_freq,
        args.force_tau,
    )
    print(
        f"[haptics] force_gain={force_gain:.4f}  "
        f"k_handle={feel['k_handle_n_per_m']:.0f} N/m  "
        f"passivity limit={feel['passivity_limit_n_per_m']:.0f} N/m  "
        f"margin={feel['margin']:.1f}x"
    )
    if feel["margin"] < 2.0:
        print(
            "[haptics] WARNING: margin below 2x. Lower --stiffness, shorten "
            "--force-tau or raise --damping before running with a device."
        )
    print(
        "[haptics] this task's translational contact force is ~1 N while carrying "
        "with ~20 N transients, so most of what there is to feel is on the grip "
        "axis. See README_conveyor.md."
    )

    device = None
    if not args.dry_run:
        from fd_omega import FDOmega

        device = FDOmega(
            poll_hz=args.control_freq,
            max_force=args.max_handle_force,
            damping_b=args.damping,
            home_pos=args.home,
        )
        device.open()

    recorder = None
    if args.dataset is not None:
        from pyrite_recorder import CONVEYOR_SCHEMA_NAME, PyriteEpisodeRecorder

        recorder = PyriteEpisodeRecorder(
            args.dataset,
            sample_hz=args.dataset_hz,
            image_size=tuple(args.image_size),
            include_rgb=args.rgb,
            schema_name=CONVEYOR_SCHEMA_NAME,
        )
        print(f"[dataset] appending to {recorder.dataset_path}")

    env = ConveyorTeleop(**env_kwargs)
    render = None
    if recorder is not None and args.rgb:
        render = env.make_camera(
            width=args.image_size[0],
            height=args.image_size[1],
            camera=args.camera,
        )

    operator = (
        ScriptedOperator(env)
        if args.dry_run
        else DeviceOperator(
            device, position_map, args.scale, env.tool_home, args.button_gripper
        )
    )
    renderer = ForceRenderer(
        force_gain, args.force_tau, args.force_rate, args.max_handle_force,
        1.0 / args.control_freq,
    )

    substeps = max(1, int(round((1.0 / args.control_freq) / env.timestep)))
    kept = 0
    try:
        for episode in range(args.episodes):
            # Index the episode explicitly rather than letting reset() advance,
            # so episode N of a session is always the same randomized episode.
            env.reset(episode_index=episode)
            operator.reset()
            if recorder is not None:
                recorder.start_episode(env.episode_metadata())
            print(
                f"[episode {episode}] belt={env.conveyor_speed_m_per_s:.3f} m/s "
                f"offset=({env.layout_offset_xy[0]:+.3f}, "
                f"{env.layout_offset_xy[1]:+.3f}) m"
            )

            target = env.tool_home.copy()
            wall_start = time.monotonic()
            ticks = int(args.max_episode_seconds * args.control_freq)
            finished = False
            for tick in range(ticks):
                command, gripper, operator_done = operator()
                if args.dry_run:
                    target = np.asarray(command, dtype=float)
                    width = float(gripper)
                else:
                    target = np.asarray(command, dtype=float)
                    width = env.gripper_width_from_fraction(gripper)
                env.step(target, n_substeps=substeps, gripper_width=width)

                sim_force = env.reflected_force(args.force_source, target)
                if device is not None:
                    device.set_reflected_force(renderer(position_map.T @ sim_force))
                    device.set_grip_force(
                        min(
                            args.grip_force_max,
                            args.grip_force_max
                            * env.grip_force()
                            / max(args.grip_force_full_scale, 1e-9),
                        )
                    )

                if recorder is not None and tick % args.record_stride == 0:
                    recorder.record_sample(
                        env,
                        timestamp_ms=1000.0 * tick / args.control_freq,
                        target_pos=target,
                        target_rotvec=None,
                        device_state=operator.state,
                        sent_force=renderer.value,
                        image_rgb=None if render is None else render(),
                        image_capture_time_s=env.current_time,
                    )

                if operator_done:
                    finished = True
                    break
                if args.auto_finish and env.judge.done(env):
                    finished = True
                    break
                if args.realtime:
                    elapsed = time.monotonic() - wall_start
                    ahead = (tick + 1) / args.control_freq - elapsed
                    if ahead > 0.0:
                        time.sleep(ahead)

            print(
                f"[episode {episode}] {'ended' if finished else 'timed out'}: "
                f"{env.termination_reason}, success={env.success()}"
            )
            if recorder is not None:
                if args.auto_keep:
                    name = recorder.commit(
                        success=env.success(),
                        termination_reason=env.termination_reason,
                        episode_attrs={
                            "conveyor_speed_m_per_s": env.conveyor_speed_m_per_s,
                            "wall_seconds": time.monotonic() - wall_start,
                        },
                    )
                    if name is None:
                        print("[dataset] too few samples, discarded")
                    else:
                        kept += 1
                        print(f"[dataset] committed {name}")
                else:
                    answer = input("keep this episode? [y/N] ").strip().lower()
                    if answer.startswith("y"):
                        name = recorder.commit(
                            success=env.success(),
                            termination_reason=env.termination_reason,
                            episode_attrs={
                                "conveyor_speed_m_per_s": env.conveyor_speed_m_per_s
                            },
                        )
                        kept += 1 if name else 0
                        print(f"[dataset] committed {name}")
                    else:
                        recorder.discard()
                        print("[dataset] discarded")
    except KeyboardInterrupt:
        print("\n[interrupt] discarding the episode in progress")
        if recorder is not None and recorder.active:
            recorder.discard()
    finally:
        env.close()
        if device is not None:
            device.close()

    if recorder is not None:
        print(f"[dataset] {kept} episode(s) written to {recorder.dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
