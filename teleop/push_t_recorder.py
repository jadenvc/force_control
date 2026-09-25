"""Full-rate episode recording for the 2D Push-T teleop task.

Mirrors ``pyrite_recorder.PyriteEpisodeRecorder``'s on-disk shape (per-episode
Zarr groups under ``data/``, an ``episode_*_len`` index under ``meta/``,
atomic write-then-move commit, in-memory ``discard()``) and its public call
pattern (``start_episode`` / ``record_sample`` / ``commit`` / ``discard``),
but with a schema that actually matches push-T's state instead of forcing
the gripper/wrench-at-tool-origin shape onto a task that has neither a
gripper nor a 6-DoF tool. See README_push_t.md for what's intentionally
NOT here yet (RGB frames -- the viewer's camera render isn't captured into
the dataset -- and adaptive-compliance virtual-target labels, which are
specific to FlipUp/cube-lift's impedance-controlled tool).

``wrench_0`` is the BC-facing signal (raw, or the causal two-pole sensor
model if the env was built with ``force_sensor_cutoff_hz`` set -- see
PushTTeleop.sensor_force_xy); ``wrench_ground_truth_0`` is always the raw
solver contact force, same split as flip-up's recorder.

``t_disturbance_wrench_0`` is ground truth for the optional unforced random
push/spin applied directly to the T (see PushTTeleop.t_disturbance_wrench,
--t-disturbance-force/-torque) -- all zero unless that feature is enabled,
and not something an operator/policy could have anticipated from the
observation stream, only useful for offline analysis of what actually moved
the T versus what the pusher did.

    data/episode_N/{ts_pose_command_0, ts_pose_controller_0, ts_pose_fb_0,
                    pusher_vel_0, t_pose_0, t_twist_0, wrench_0,
                    wrench_ground_truth_0, t_disturbance_wrench_0, ...}
    meta/episode_robot0_len
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import mujoco
import numpy as np

from pyrite_recorder import _NumericSampleBuffer, _jsonable, _zarr_modules

SCHEMA_NAME = "pyrite_push_t_sim"
SCHEMA_VERSION = 1
DEFAULT_SAMPLE_HZ = 1000.0


class PushTEpisodeRecorder:
    """Accumulate one push-T episode in memory and atomically append it to Zarr.

    Same start/record/commit/discard contract as ``PyriteEpisodeRecorder``,
    scoped to push-T's actual state instead of a gripper/tool schema.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        sample_hz: float = DEFAULT_SAMPLE_HZ,
        min_samples: int = 20,
    ) -> None:
        if sample_hz <= 0.0:
            raise ValueError("sample_hz must be positive")
        self.zarr, self.numcodecs = _zarr_modules()
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_hz = float(sample_hz)
        self.min_samples = int(min_samples)

        self.root = self.zarr.open(str(self.dataset_path), mode="a")
        self.data_group = self.root.require_group("data")
        self.meta_group = self.root.require_group("meta")
        if "schema_name" in self.root.attrs:
            if self.root.attrs["schema_name"] != SCHEMA_NAME:
                raise RuntimeError(
                    f"{self.dataset_path} uses schema "
                    f"{self.root.attrs['schema_name']!r}, expected {SCHEMA_NAME!r}"
                )
            old_hz = float(self.root.attrs["sample_hz"])
            if not np.isclose(old_hz, self.sample_hz):
                raise RuntimeError(
                    f"dataset rate is {old_hz:g} Hz, requested {self.sample_hz:g} Hz"
                )
        else:
            self.root.attrs.update(
                {
                    "schema_name": SCHEMA_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "sample_hz": self.sample_hz,
                    "timestamp_unit": "milliseconds",
                    "wrench_convention": "force_xy_on_pusher_from_t_block",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
        self._buffer_capacity = max(1024, int(round(20.0 * self.sample_hz)))
        self._samples = _NumericSampleBuffer(self._buffer_capacity)
        self._metadata: dict[str, Any] = {}
        self._started = False
        self._state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self._state_size: int | None = None

    @property
    def sample_count(self) -> int:
        return self._samples.length("robot_time_stamps_0")

    @property
    def active(self) -> bool:
        return self._started

    @property
    def episode_names(self) -> list[str]:
        return sorted(
            (
                key
                for key in self.data_group.group_keys()
                if key.startswith("episode_")
            ),
            key=lambda key: int(key.rsplit("_", 1)[-1]),
        )

    def start_episode(self, metadata: dict[str, Any] | None = None) -> None:
        if self.active:
            raise RuntimeError("finish or discard the active episode first")
        self._samples = _NumericSampleBuffer(self._buffer_capacity)
        self._metadata = _jsonable(metadata or {})
        self._started = True

    def record_sample(
        self,
        env,
        *,
        timestamp_ms: float,
        target_xy: np.ndarray,
        device_state: dict[str, Any] | None,
        sent_force: np.ndarray,
        wall_time_ns: int | None = None,
    ) -> bool:
        """Capture one control-rate sample of push-T state."""
        if not self._started:
            self.start_episode()

        if self._state_size is None:
            self._state_size = mujoco.mj_stateSize(env.model.ptr, self._state_spec)
        integration_state = np.empty(self._state_size, dtype=np.float64)
        mujoco.mj_getState(
            env.model.ptr,
            env.data.ptr,
            integration_state,
            self._state_spec,
        )

        state = device_state or {}
        self._samples.append("robot_time_stamps_0", float(timestamp_ms))
        self._samples.append(
            "wall_time_ns", 0 if wall_time_ns is None else int(wall_time_ns)
        )
        # Keep the operator's raw requested target separate from what
        # actually drove the spring: they differ while the workspace clamp
        # (limited_target) is active, same reasoning as flip-up's
        # ts_pose_command_0 / ts_pose_controller_0 split.
        self._samples.append("ts_pose_command_0", np.asarray(target_xy, dtype=float))
        self._samples.append("ts_pose_controller_0", env.drive_target)
        self._samples.append("ts_pose_fb_0", env.pusher_pos)
        self._samples.append("pusher_vel_0", env.pusher_vel)
        self._samples.append("t_pose_0", env.t_pose)
        self._samples.append("t_twist_0", env.t_twist)
        # wrench_0 is the BC-facing signal: the causal finite-bandwidth
        # sensor model if --force-sensor-cutoff is set (env.sensor_force_xy()
        # equals the raw value when it isn't), matching flip-up's wrench_0 /
        # wrench_ground_truth_0 split -- see PushTTeleop.sensor_force_xy for
        # why the raw signal is discretely noisy even with a smooth target.
        self._samples.append("wrench_0", env.sensor_force_xy())
        self._samples.append("wrench_ground_truth_0", env.pusher_contact_force_xy())
        # Ground truth only -- an operator/policy can't see this coming (it's
        # an unforced random process, not a scripted path), but recording it
        # lets offline analysis separate "the T moved on its own" from "the
        # push did that". All zero unless --t-disturbance-force/-torque set.
        self._samples.append("t_disturbance_wrench_0", env.t_disturbance_wrench)
        self._samples.append("contact_count", int(env.data.ncon))
        self._samples.append("sim_time_s", float(env.data.time))
        self._samples.append("success", int(env.success()))
        self._samples.append("coverage_fraction", float(env.coverage_fraction()))
        self._samples.append("task_metric", float(env.coverage_fraction()))
        self._samples.append("qpos", env.data.qpos)
        self._samples.append("qvel", env.data.qvel)
        self._samples.append("qacc", env.data.qacc)
        self._samples.append("mujoco_state", integration_state)
        self._samples.append("device_pos", state.get("pos", np.zeros(3)))
        self._samples.append("device_vel", state.get("vel", np.zeros(3)))
        self._samples.append("device_force_cmd", state.get("force_cmd", np.zeros(3)))
        self._samples.append(
            "device_force_measured", state.get("force_meas", np.zeros(3))
        )
        self._samples.append(
            "device_long_press_count", int(state.get("long_press_count", 0))
        )
        self._samples.append(
            "device_short_press_count", int(state.get("short_press_count", 0))
        )
        self._samples.append("haptic_force_sent", np.asarray(sent_force, dtype=float))
        return True

    def _next_episode_id(self) -> int:
        names = self.episode_names
        return 0 if not names else max(int(n.rsplit("_", 1)[-1]) for n in names) + 1

    def _write_array(self, group, key: str, value: np.ndarray) -> None:
        compressor = self.numcodecs.Blosc(
            cname="zstd",
            clevel=3,
            shuffle=(
                self.numcodecs.Blosc.BITSHUFFLE
                if value.dtype != np.uint8
                else self.numcodecs.Blosc.SHUFFLE
            ),
        )
        chunks = (
            () if value.ndim == 0
            else (min(256, len(value)),)
            + tuple(max(1, int(dimension)) for dimension in value.shape[1:])
        )
        group.array(
            name=key, data=value, chunks=chunks, compressor=compressor, overwrite=True
        )

    def _update_meta(self) -> None:
        names = self.episode_names
        robot_lengths = [len(self.data_group[name]["ts_pose_fb_0"]) for name in names]
        self.meta_group.array(
            name="episode_robot0_len",
            data=np.asarray(robot_lengths, dtype=np.int64),
            chunks=(max(1, len(robot_lengths)),),
            compressor=None,
            overwrite=True,
        )

    def commit(
        self,
        *,
        success: bool,
        termination_reason: str,
        final_coverage_fraction: float,
        final_task_metric_name: str | None = None,
        final_task_metric_value: float | None = None,
    ) -> str | None:
        count = self.sample_count
        if count < self.min_samples:
            self.discard()
            return None
        arrays = self._samples.arrays()

        episode_id = self._next_episode_id()
        name = f"episode_{episode_id}"
        temp_name = f"_episode_{episode_id}_writing"
        if temp_name in self.data_group:
            del self.data_group[temp_name]
        episode = self.data_group.create_group(temp_name)
        try:
            for key, value in arrays.items():
                self._write_array(episode, key, value)
            episode.attrs.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_hz": self.sample_hz,
                    "success": bool(success),
                    "termination_reason": str(termination_reason),
                    "final_coverage_fraction": float(final_coverage_fraction),
                    "final_task_metric_name": (
                        "coverage_fraction"
                        if final_task_metric_name is None
                        else str(final_task_metric_name)
                    ),
                    "final_task_metric_value": float(
                        final_coverage_fraction
                        if final_task_metric_value is None
                        else final_task_metric_value
                    ),
                    "sample_count": count,
                    "mujoco_state_spec": self._state_spec,
                    "metadata_json": json.dumps(self._metadata, sort_keys=True),
                    "committed_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            self.data_group.move(temp_name, name)
            self._update_meta()
        except Exception:
            if temp_name in self.data_group:
                del self.data_group[temp_name]
            raise
        finally:
            self._samples = _NumericSampleBuffer(self._buffer_capacity)
            self._metadata = {}
            self._started = False
        return name

    def discard(self) -> int:
        count = self.sample_count
        self._samples = _NumericSampleBuffer(self._buffer_capacity)
        self._metadata = {}
        self._started = False
        return count
