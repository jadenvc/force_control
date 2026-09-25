"""Cheap, replay-friendly episode recording for FlipUp teleoperation.

Sibling of ``pyrite_recorder.PyriteEpisodeRecorder``, but deliberately skips
every per-tick MuJoCo introspection call that's expensive under load:
``mj_getState`` (full integration-state copy), the contact/wrist/sensor
wrench extraction (a per-contact loop, more costly with the 3-5 simultaneous
contacts a real flip produces), and the raw qpos/qvel/qacc/ctrl/sensordata
array copies. See LATENCY_DIAGNOSTICS.md for the measurement that motivated
this -- in one smoke test, the FULL recorder's per-tick cost (mostly those
calls) exceeded the physics step's own cost.

What's kept is exactly what's needed to (a) faithfully REPLAY the episode
later through ``replay_flipup_sparse.py`` to reconstruct the full dense
state/wrench stream offline (where there's no real-time deadline, so the
heavy calls' cost doesn't matter), and (b) the live device-side telemetry
that replay can never reconstruct, because it's a property of the physical
haptic device in that live loop, not a function of the simulated state:

    - ts_pose_command_0 / target_rotvec: the operator's commanded target --
      literally what a replay drives the arm with.
    - ts_pose_fb_0: the achieved tool pose (cheap site xpos/xmat reads, no
      state copy) -- useful for a live tracking-lag check without needing to
      touch anything heavier.
    - device_*: position/velocity/orientation/gripper/force_cmd/
      force_measured/servo timing straight from the Force Dimension SDK.
    - haptic_force_sent, gripper_command.
    - control_batch_size/index, deadline_lateness_ms, and the diag_* latency
      fields (see teleop_flipup.py) -- so a sparse-mode run still carries its
      own confirmation of whether it actually achieved low latency.

No RGB support on purpose: the async renderer is already decoupled/cheap
relative to this, but a sparse recording is meant to be as small and fast as
possible, and RGB can always be captured fresh during a later replay pass
(which re-renders every frame anyway) instead of live.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import platform
import socket

import numpy as np

from pyrite_recorder import _NumericSampleBuffer, _jsonable, _zarr_modules


SCHEMA_NAME = "pyrite_flipup_sparse"
SCHEMA_VERSION = 1
DEFAULT_SAMPLE_HZ = 1000.0


class SparseFlipUpRecorder:
    """Accumulate one episode's cheap command/device stream, write to Zarr."""

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
        else:
            self.root.attrs.update(
                {
                    "schema_name": SCHEMA_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "sample_hz": self.sample_hz,
                    "timestamp_unit": "milliseconds",
                    "pose_convention": "xyz+wxyz",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "host": socket.gethostname(),
                    "platform": platform.platform(),
                }
            )
        self._buffer_capacity = max(1024, int(round(20.0 * self.sample_hz)))
        self._samples = _NumericSampleBuffer(self._buffer_capacity)
        self._metadata: dict[str, Any] = {}
        self._started = False

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

    def _append(self, key: str, value: Any) -> None:
        self._samples.append(key, value)

    def record_sample(
        self,
        env,
        *,
        timestamp_ms: float,
        target_pos: np.ndarray,
        target_rotvec: np.ndarray | None,
        device_state: dict[str, Any] | None,
        sent_force: np.ndarray,
        wall_time_ns: int | None = None,
        control_batch_size: int = 1,
        control_batch_index: int = 0,
        deadline_lateness_ms: float = 0.0,
        diag_prev_iter_ms: float = 0.0,
        diag_command_ms: float = 0.0,
        diag_step_ms: float = 0.0,
        diag_record_ms: float = 0.0,
        diag_catchup_debt_ticks: float = 0.0,
        diag_prev_view_ms: float = 0.0,
        diag_prev_sleep_ms: float = 0.0,
    ) -> bool:
        """Capture one control sample -- cheap fields only, see module docstring."""
        if not self._started:
            self.start_episode()

        command_pose = env.target_pose7(target_pos, target_rotvec)
        feedback_pose = np.concatenate([env.tool_pos, env.tool_quat])
        state = device_state or {}

        self._append("robot_time_stamps_0", float(timestamp_ms))
        self._append("wall_time_ns", 0 if wall_time_ns is None else int(wall_time_ns))
        self._append("control_batch_size", int(control_batch_size))
        self._append("control_batch_index", int(control_batch_index))
        self._append("deadline_lateness_ms", float(deadline_lateness_ms))
        self._append("diag_prev_iter_ms", float(diag_prev_iter_ms))
        self._append("diag_command_ms", float(diag_command_ms))
        self._append("diag_step_ms", float(diag_step_ms))
        self._append("diag_record_ms", float(diag_record_ms))
        self._append("diag_catchup_debt_ticks", float(diag_catchup_debt_ticks))
        self._append("diag_prev_view_ms", float(diag_prev_view_ms))
        self._append("diag_prev_sleep_ms", float(diag_prev_sleep_ms))
        self._append("ts_pose_command_0", command_pose)
        self._append("ts_pose_fb_0", feedback_pose)
        self._append("target_rotvec", np.zeros(3) if target_rotvec is None else target_rotvec)
        self._append("gripper_command", float(getattr(env, "gripper_command", 0.0)))
        self._append("device_pos", state.get("pos", np.zeros(3)))
        self._append("device_vel", state.get("vel", np.zeros(3)))
        self._append("device_rotmat", state.get("rot", np.eye(3)))
        self._append("device_gripper", state.get("gripper", 0.0))
        self._append("device_grip_force_target", state.get("grip_force_target", 0.0))
        self._append("device_grip_force_applied", state.get("grip_force_applied", 0.0))
        self._append("device_force_cmd", state.get("force_cmd", np.zeros(3)))
        self._append("device_force_measured", state.get("force_meas", np.zeros(3)))
        self._append("device_orientation_valid", int(state.get("orientation_valid", False)))
        self._append("device_servo_sequence", int(state.get("servo_sequence", -1)))
        self._append("device_servo_timestamp_ns", int(state.get("servo_timestamp_ns", 0)))
        self._append("device_servo_dt_s", float(state.get("servo_dt_s", 0.0)))
        self._append("haptic_force_sent", sent_force)
        return True

    def _next_episode_id(self) -> int:
        names = self.episode_names
        return 0 if not names else max(int(n.rsplit("_", 1)[-1]) for n in names) + 1

    def _write_array(self, group, key: str, value: np.ndarray) -> None:
        compressor = self.numcodecs.Blosc(
            cname="zstd", clevel=3,
            shuffle=self.numcodecs.Blosc.BITSHUFFLE,
        )
        chunks = (
            ()
            if value.ndim == 0
            else (min(256, len(value)),) + tuple(
                max(1, int(dimension)) for dimension in value.shape[1:]
            )
        )
        group.array(
            name=key, data=value, chunks=chunks, compressor=compressor, overwrite=True,
        )

    def _update_meta(self) -> None:
        names = self.episode_names
        lengths = []
        for name in names:
            episode = self.data_group[name]
            lengths.append(len(episode["ts_pose_fb_0"]))
        self.meta_group.array(
            name="episode_robot0_len",
            data=np.asarray(lengths, dtype=np.int64),
            chunks=(max(1, len(lengths)),),
            compressor=None,
            overwrite=True,
        )

    def commit(
        self,
        *,
        success: bool,
        termination_reason: str,
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
                    "final_task_metric_name": (
                        "" if final_task_metric_name is None else str(final_task_metric_name)
                    ),
                    "final_task_metric_value": float(final_task_metric_value or 0.0),
                    "sample_count": count,
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
