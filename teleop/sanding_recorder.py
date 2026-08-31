"""Pyrite-style episode recording for the sanding teleop task.

Deliberately a sibling of ``pyrite_recorder.py``, not an edit to it in place.
``PyriteEpisodeRecorder.record_sample``/``commit`` hardcode FlipUp fields
directly in their bodies (``env.book_angle_deg()``, ``env.surface_limit_active``,
a required ``final_book_angle_deg`` argument, ...) rather than accepting a
generic per-step field dict, so bolting sanding fields into that same method
risks breaking existing FlipUp datasets/the standalone validator. This module
reuses the parts of ``pyrite_recorder.py`` that ARE generic -- the growable
per-key sample buffer and the JSON-metadata helper -- and reimplements
``record_sample``/``commit`` against ``SandingEnv`` instead of ``FlipUpEnv``.

On-disk layout (same shape as pyrite_recorder.py's, different schema name so
sanding datasets never collide with FlipUp ones on disk):

    data/episode_N/{rgb_0, ts_pose_fb_0, ts_pose_command_0, wrench_0,
                    wrench_ground_truth_0, coverage_under, coverage_just_right,
                    coverage_over, dose_mean, dose_max, broken, success, ...}
    meta/{episode_rgb0_len, episode_robot0_len, episode_wrench0_len}

The full per-cell dose grid (~1200 floats/step at 1kHz would be ~288MB per
30s episode) is intentionally NOT recorded per-step -- a BC policy would
condition on the compact coverage/dose summary fields above, not the raw
grid. See README_sanding.md's "Not yet implemented" section for an opt-in,
heavily-decimated full-grid snapshot as a possible follow-up.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import numpy as np

from pyrite_recorder import _NumericSampleBuffer, _jsonable, _zarr_modules

SCHEMA_NAME = "pyrite_sanding_sim"
SCHEMA_VERSION = 1
DEFAULT_SAMPLE_HZ = 1000.0


class SandingEpisodeRecorder:
    """Accumulate one sanding episode in memory and atomically append it to Zarr."""

    def __init__(
        self,
        dataset_path,
        *,
        sample_hz: float = DEFAULT_SAMPLE_HZ,
        image_size: tuple[int, int] = (224, 224),
        include_rgb: bool = True,
        min_samples: int = 20,
    ) -> None:
        if sample_hz <= 0.0:
            raise ValueError("sample_hz must be positive")
        if any(int(v) <= 0 for v in image_size):
            raise ValueError("image_size must be positive")
        self.zarr, self.numcodecs = _zarr_modules()
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_hz = float(sample_hz)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.include_rgb = bool(include_rgb)
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
                    "pose_convention": "xyz+wxyz",
                    "wrench_convention": "force_xyz+torque_xyz_at_tool_origin",
                    "wrench_frame": "tool",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
        self._buffer_capacity = max(1024, int(round(20.0 * self.sample_hz)))
        self._samples = _NumericSampleBuffer(self._buffer_capacity)
        self._images: list[np.ndarray] = []
        self._image_timestamps_ms: list[float] = []
        self._last_image_id: int | None = None
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
            (k for k in self.data_group.group_keys() if k.startswith("episode_")),
            key=lambda key: int(key.rsplit("_", 1)[-1]),
        )

    def start_episode(self, metadata: dict[str, Any] | None = None) -> None:
        if self.active:
            raise RuntimeError("finish or discard the active episode first")
        self._samples = _NumericSampleBuffer(self._buffer_capacity)
        self._images = []
        self._image_timestamps_ms = []
        self._last_image_id = None
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
        image_rgb: np.ndarray | None,
        image_capture_time_s: float | None = None,
        image_id: int | None = None,
        wall_time_ns: int | None = None,
        control_batch_size: int = 1,
        control_batch_index: int = 0,
        deadline_lateness_ms: float = 0.0,
    ) -> bool:
        """Capture one control sample and, when new, one asynchronous RGB frame."""
        if not self._started:
            self.start_episode()

        target_quat = env.target_pose7(target_pos, target_rotvec)[3:]
        command_pose = np.concatenate([np.asarray(target_pos, dtype=float), target_quat])
        feedback_pose = env.get_tool_pose().astype(np.float64)

        wrench_force, _ = env.pad_contact_force()
        wrench = np.concatenate([wrench_force, np.zeros(3)])  # no torque sensor modeled

        state = device_state or {}
        self._append("robot_time_stamps_0", float(timestamp_ms))
        self._append("wrench_time_stamps_0", float(timestamp_ms))
        self._append("wall_time_ns", 0 if wall_time_ns is None else int(wall_time_ns))
        self._append("control_batch_size", int(control_batch_size))
        self._append("control_batch_index", int(control_batch_index))
        self._append("deadline_lateness_ms", float(deadline_lateness_ms))
        self._append("ts_pose_fb_0", feedback_pose)
        self._append("ts_pose_command_0", command_pose)
        self._append("wrench_0", wrench)
        self._append("wrench_ground_truth_0", wrench)
        self._append("normal_force_n", env.normal_force_n())
        self._append("coverage_under", env.coverage_fraction("under"))
        self._append("coverage_just_right", env.coverage_fraction("just_right"))
        self._append("coverage_over", env.coverage_fraction("over"))
        self._append("dose_mean", float(env._dose.mean()))
        self._append("dose_max", float(env._dose.max()))
        self._append("broken", int(env.broken))
        self._append("success", int(env.success()))
        self._append("contact_count", int(env.data.ncon))
        self._append("sim_time_s", float(env.data.time))
        self._append("qpos", env.data.qpos[env.joint_qpos_ids])
        self._append("qvel", env.data.qvel[env.joint_dof_ids])
        self._append("ctrl", env.data.ctrl[env.actuator_ids])
        self._append("target_rotvec", np.zeros(3) if target_rotvec is None else target_rotvec)
        self._append("device_pos", state.get("pos", np.zeros(3)))
        self._append("device_vel", state.get("vel", np.zeros(3)))
        self._append("device_force_cmd", state.get("force_cmd", np.zeros(3)))
        self._append("device_force_measured", state.get("force_meas", np.zeros(3)))
        self._append("haptic_force_sent", sent_force)
        self._append(
            "rgb_capture_sim_time_s",
            np.nan if image_capture_time_s is None else image_capture_time_s,
        )

        new_image = (
            self.include_rgb
            and image_rgb is not None
            and (image_id is None or image_id != self._last_image_id)
        )
        if new_image:
            import cv2

            image = np.asarray(image_rgb, dtype=np.uint8)
            width, height = self.image_size
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            self._images.append(image.copy())
            capture_timestamp_ms = (
                float(timestamp_ms)
                if image_capture_time_s is None or not np.isfinite(image_capture_time_s)
                else max(0.0, 1000.0 * float(image_capture_time_s))
            )
            self._image_timestamps_ms.append(capture_timestamp_ms)
            self._last_image_id = image_id
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
        if value.ndim == 0:
            chunks = ()
        elif key == "rgb_0":
            chunks = (1,) + value.shape[1:]
        else:
            chunks = (min(256, len(value)),) + tuple(
                max(1, int(dimension)) for dimension in value.shape[1:]
            )
        group.array(name=key, data=value, chunks=chunks, compressor=compressor, overwrite=True)

    def _update_meta(self) -> None:
        names = self.episode_names
        rgb_lengths, robot_lengths, wrench_lengths = [], [], []
        for name in names:
            episode = self.data_group[name]
            rgb_lengths.append(len(episode["rgb_0"]))
            robot_lengths.append(len(episode["ts_pose_fb_0"]))
            wrench_lengths.append(len(episode["wrench_0"]))
        for key, values in (
            ("episode_rgb0_len", rgb_lengths),
            ("episode_robot0_len", robot_lengths),
            ("episode_wrench0_len", wrench_lengths),
        ):
            self.meta_group.array(
                name=key,
                data=np.asarray(values, dtype=np.int64),
                chunks=(max(1, len(values)),),
                compressor=None,
                overwrite=True,
            )

    def commit(
        self,
        *,
        success: bool,
        broken: bool,
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
        if self.include_rgb and self._images:
            arrays["rgb_0"] = np.asarray(self._images, dtype=np.uint8)
            arrays["rgb_time_stamps_0"] = np.asarray(self._image_timestamps_ms, dtype=np.float64)
        else:
            width, height = self.image_size
            arrays["rgb_0"] = np.zeros((1, height, width, 3), dtype=np.uint8)
            arrays["rgb_time_stamps_0"] = np.zeros(1, dtype=np.float64)

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
                    "broken": bool(broken),
                    "termination_reason": str(termination_reason),
                    "final_coverage_fraction": float(final_coverage_fraction),
                    "final_task_metric_name": (
                        "coverage_just_right"
                        if final_task_metric_name is None
                        else str(final_task_metric_name)
                    ),
                    "final_task_metric_value": float(
                        final_coverage_fraction
                        if final_task_metric_value is None
                        else final_task_metric_value
                    ),
                    "sample_count": count,
                    "rgb_sample_count": len(arrays["rgb_0"]),
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
            self._images = []
            self._image_timestamps_ms = []
            self._last_image_id = None
            self._metadata = {}
            self._started = False
        return name

    def discard(self) -> int:
        count = self.sample_count
        self._samples = _NumericSampleBuffer(self._buffer_capacity)
        self._images = []
        self._image_timestamps_ms = []
        self._last_image_id = None
        self._metadata = {}
        self._started = False
        return count
