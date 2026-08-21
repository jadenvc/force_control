"""Full-rate Pyrite-compatible episode recording for FlipUp teleoperation.

The on-disk layout matches PyriteML's current ``ReplayBuffer`` and
``VirtualTargetDataset`` contracts:

    data/episode_N/{rgb_0, ts_pose_fb_0, ts_pose_command_0,
                    ts_pose_virtual_target_0, stiffness_0, wrench_0, ...}
    meta/{episode_rgb0_len, episode_robot0_len, episode_wrench0_len}

Robot state, command, wrench and complete MuJoCo integration state are stored at
the control rate (1 kHz by default). RGB is stored only when the asynchronous
renderer produces a new frame, with independent timestamps; duplicating a
30 Hz image at 1 kHz would waste memory without adding information.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import platform
import socket

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA_NAME = "pyrite_flipup_sim"
SCHEMA_VERSION = 4
DEFAULT_SAMPLE_HZ = 1000.0
PYRITE_REQUIRED_KEYS = (
    "rgb_0",
    "rgb_time_stamps_0",
    "ts_pose_fb_0",
    "ts_pose_command_0",
    "ts_pose_virtual_target_0",
    "stiffness_0",
    "wrench_0",
    "robot_time_stamps_0",
    "wrench_time_stamps_0",
)


class _NumericSampleBuffer:
    """Growable contiguous arrays without one Python object per 1 kHz field.

    The old list-of-small-arrays layout created hundreds of thousands of Python
    and NumPy objects during a demonstration. That work ran in the control loop
    and was a major source of 16-tick catch-up bursts. Allocation happens once
    per channel, normally in sample zero before live control resumes.
    """

    def __init__(self, initial_capacity: int) -> None:
        self.initial_capacity = max(1, int(initial_capacity))
        self._storage: dict[str, np.ndarray] = {}
        self._lengths: dict[str, int] = {}

    def append(self, key: str, value: Any) -> None:
        value_array = np.asarray(value)
        if key not in self._storage:
            self._storage[key] = np.empty(
                (self.initial_capacity,) + value_array.shape,
                dtype=value_array.dtype,
            )
            self._lengths[key] = 0
        storage = self._storage[key]
        length = self._lengths[key]
        if value_array.shape != storage.shape[1:]:
            raise ValueError(
                f"sample shape for {key!r} changed from {storage.shape[1:]} "
                f"to {value_array.shape}"
            )
        if length >= len(storage):
            grown = np.empty((2 * len(storage),) + storage.shape[1:], storage.dtype)
            grown[:length] = storage
            storage = self._storage[key] = grown
        storage[length] = value_array
        self._lengths[key] = length + 1

    def length(self, key: str) -> int:
        return self._lengths.get(key, 0)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            key: storage[: self._lengths[key]]
            for key, storage in self._storage.items()
        }


def _zarr_modules():
    try:
        import numcodecs
        import zarr
    except ImportError as exc:
        raise RuntimeError(
            "Dataset collection needs zarr 2.x and numcodecs. Install with "
            "`python -m pip install -r teleop/requirements_dataset.txt`."
        ) from exc
    major = int(zarr.__version__.split(".", 1)[0])
    if major >= 3:
        raise RuntimeError(
            f"PyriteML requires zarr 2.x, but zarr {zarr.__version__} is installed"
        )
    return zarr, numcodecs


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _causal_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    cumulative = np.concatenate(
        [np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)],
        axis=0,
    )
    output = np.empty_like(values)
    for i in range(len(values)):
        start = max(0, i + 1 - window)
        output[i] = (cumulative[i + 1] - cumulative[start]) / (i + 1 - start)
    return output


def adaptive_compliance_labels(
    command_pose7: np.ndarray,
    wrench_tool: np.ndarray,
    *,
    k_max: float,
    k_min: float,
    f_low: float,
    f_high: float,
    dim: int = 3,
    characteristic_length: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Pyrite's virtual-target and scalar-stiffness action labels.

    This is the vectorized ``VirtualTargetEstimator`` rule used by
    PyriteUtility. ``wrench_tool`` is the wrench the world applies to the robot;
    Pyrite negates it to obtain the commanded compliance displacement.
    """
    poses = np.asarray(command_pose7, dtype=np.float64)
    wrench = np.asarray(wrench_tool, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"command_pose7 must have shape (T, 7), got {poses.shape}")
    if wrench.shape != (len(poses), 6):
        raise ValueError(f"wrench_tool must have shape {(len(poses), 6)}, got {wrench.shape}")
    if dim not in (3, 6):
        raise ValueError("adaptive compliance dimension must be 3 or 6")
    if not (0.0 < k_min <= k_max):
        raise ValueError("adaptive stiffness must satisfy 0 < k_min <= k_max")
    if not (0.0 <= f_low < f_high):
        raise ValueError("adaptive force thresholds must satisfy 0 <= low < high")
    if characteristic_length <= 0.0:
        raise ValueError("characteristic_length must be positive")

    generalized_force = -wrench[:, :dim].copy()
    regularized_force = generalized_force.copy()
    if dim == 6:
        regularized_force[:, 3:] /= characteristic_length
    force_norm = np.linalg.norm(regularized_force, axis=1)

    stiffness = np.empty(len(poses), dtype=np.float64)
    stiffness[force_norm < f_low] = k_max
    stiffness[force_norm > f_high] = k_min
    blend = (force_norm >= f_low) & (force_norm <= f_high)
    stiffness[blend] = k_max - (k_max - k_min) * (
        force_norm[blend] - f_low
    ) / (f_high - f_low)

    displacement = np.zeros_like(regularized_force)
    loaded = force_norm >= f_low
    displacement[loaded] = (
        regularized_force[loaded] / stiffness[loaded, np.newaxis]
    )
    if dim == 6:
        displacement[:, 3:] /= characteristic_length

    rotations = Rotation.from_quat(poses[:, [4, 5, 6, 3]])
    virtual = poses.copy()
    virtual[:, :3] += rotations.apply(displacement[:, :3])
    if dim == 6:
        delta_rotation = Rotation.from_rotvec(displacement[:, 3:])
        virtual_rotation = rotations * delta_rotation
        virtual[:, 3:] = virtual_rotation.as_quat()[:, [3, 0, 1, 2]]
    return virtual, stiffness


class PyriteEpisodeRecorder:
    """Accumulate one episode in memory and atomically append it to Zarr."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        sample_hz: float = DEFAULT_SAMPLE_HZ,
        image_size: tuple[int, int] = (224, 224),
        include_rgb: bool = True,
        min_samples: int = 20,
        wrench_filter_seconds: float = 0.25,
        ac_k_max: float = 16000.0,
        ac_k_min: float = 2000.0,
        ac_f_low: float = 2.0,
        ac_f_high: float = 100.0,
        ac_dim: int = 3,
        ac_characteristic_length: float = 0.02,
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
        self.wrench_filter_samples = max(
            1, int(round(float(wrench_filter_seconds) * self.sample_hz))
        )
        self.adaptive_config = {
            "k_max": float(ac_k_max),
            "k_min": float(ac_k_min),
            "f_low": float(ac_f_low),
            "f_high": float(ac_f_high),
            "dim": int(ac_dim),
            "characteristic_length": float(ac_characteristic_length),
        }

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
            old_schema_version = int(self.root.attrs.get("schema_version", 1))
            if old_schema_version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"dataset schema {old_schema_version} is newer than supported "
                    f"schema {SCHEMA_VERSION}"
                )
            if old_schema_version < SCHEMA_VERSION:
                self.root.attrs["schema_version"] = SCHEMA_VERSION
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
                    "host": socket.gethostname(),
                    "platform": platform.platform(),
                }
            )
        self._buffer_capacity = max(1024, int(round(20.0 * self.sample_hz)))
        self._samples = _NumericSampleBuffer(self._buffer_capacity)
        self._images: list[np.ndarray] = []
        self._image_timestamps_ms: list[float] = []
        self._last_image_id: int | None = None
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

        command_pose = env.target_pose7(target_pos, target_rotvec)
        controller_target = np.asarray(
            getattr(env, "drive_target", target_pos), dtype=np.float64
        )
        controller_pose = env.target_pose7(controller_target, target_rotvec)
        feedback_pose = np.concatenate([env.tool_pos, env.tool_quat])
        if getattr(env, "controller_kind", None) == "floating_gripper":
            # Both paths are cached once per physics tick by the floating env:
            # exact solver contact remains ground truth while the optional
            # causal sensor model is the observation available to BC.
            truth_world = env.contact_wrench(frame="world")
            truth_tool = env.contact_wrench(frame="tool")
            sensed_tool = env.sensor_wrench(frame="tool")
            sensed_world = env.sensor_wrench(frame="world")
            sensor_raw = truth_tool
        else:
            sensed_tool = env.wrist_wrench(frame="tool")
            sensed_world = env.wrist_wrench(frame="world")
            sensor_raw = env.wrist_wrench_raw()
            truth_tool = env.contact_wrench(frame="tool")
            truth_world = env.contact_wrench(frame="world")

        tool_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            env.model.ptr,
            env.data.ptr,
            mujoco.mjtObj.mjOBJ_SITE,
            env.tool_site_id,
            tool_velocity,
            0,
        )
        book_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            env.model.ptr,
            env.data.ptr,
            mujoco.mjtObj.mjOBJ_BODY,
            env.book_body_id,
            book_velocity,
            0,
        )
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
        self._append("robot_time_stamps_0", float(timestamp_ms))
        self._append("wrench_time_stamps_0", float(timestamp_ms))
        self._append("wall_time_ns", 0 if wall_time_ns is None else int(wall_time_ns))
        self._append("control_batch_size", int(control_batch_size))
        self._append("control_batch_index", int(control_batch_index))
        self._append("deadline_lateness_ms", float(deadline_lateness_ms))
        self._append("ts_pose_fb_0", feedback_pose)
        self._append("ts_pose_command_0", command_pose)
        # Keep the operator's requested pose above, and separately store the
        # target actually sent to the impedance controller.  They differ only
        # while visible-surface anti-windup is active; retaining both prevents a
        # safety projection from being mistaken for demonstrated compliance.
        self._append("ts_pose_controller_0", controller_pose)
        self._append("surface_limit_active", int(env.surface_limit_active))
        self._append(
            "workspace_limit_active",
            int(getattr(env, "workspace_limit_active", False)),
        )
        self._append("wrench_0", sensed_tool)
        self._append("wrench_sensor_world_0", sensed_world)
        self._append("wrench_sensor_raw_0", sensor_raw)
        self._append("wrench_sensor_model_0", sensed_tool)
        self._append("wrench_sensor_model_world_0", sensed_world)
        self._append("wrench_ground_truth_0", truth_tool)
        self._append("wrench_ground_truth_world_0", truth_world)
        self._append("robot_wrench_0", truth_tool)
        self._append("tool_twist_world", np.r_[tool_velocity[3:], tool_velocity[:3]])
        self._append("book_pose", np.r_[env.book_pos, env.book_quat])
        self._append("object_pose", np.r_[env.book_pos, env.book_quat])
        self._append("book_twist_world", np.r_[book_velocity[3:], book_velocity[:3]])
        self._append("object_twist_world", np.r_[book_velocity[3:], book_velocity[:3]])
        self._append("book_angle_deg", env.book_angle_deg())
        self._append(
            "task_metric",
            (
                env.task_metric_value()
                if hasattr(env, "task_metric_value")
                else env.book_angle_deg()
            ),
        )
        self._append(
            "object_height_m",
            float(getattr(env, "object_height_m", env.book_pos[2])),
        )
        self._append(
            "object_lift_height_m",
            float(getattr(env, "lift_height_m", 0.0)),
        )
        self._append(
            "gripper_command",
            float(getattr(env, "gripper_command", 0.0)),
        )
        self._append(
            "gripper_controller_target",
            float(getattr(env, "gripper_controller_target", 0.0)),
        )
        self._append(
            "gripper_opening",
            float(getattr(env, "gripper_opening", 0.0)),
        )
        self._append(
            "gripper_actuator_force",
            float(getattr(env, "gripper_actuator_force", 0.0)),
        )
        grasp_force = float(
            env.grasp_force() if hasattr(env, "grasp_force") else 0.0
        )
        table_force = float(
            env.table_contact_force()
            if hasattr(env, "table_contact_force")
            else 0.0
        )
        grasp_limit = float(getattr(env, "grasp_force_limit", np.inf))
        table_limit = float(getattr(env, "surface_force_limit", np.inf))
        self._append("grasp_force", grasp_force)
        self._append("table_contact_force", table_force)
        self._append(
            "grasp_force_limit_exceeded", int(grasp_force > grasp_limit + 1e-6)
        )
        self._append(
            "table_contact_force_limit_exceeded",
            int(table_force > table_limit + 1e-6),
        )
        self._append("success", int(env.success()))
        self._append("contact_count", int(env.data.ncon))
        self._append("sim_time_s", float(env.data.time))
        self._append("qpos", env.data.qpos)
        self._append("qvel", env.data.qvel)
        self._append("qacc", env.data.qacc)
        self._append("ctrl", env.data.ctrl)
        self._append("actuator_force", env.data.actuator_force)
        self._append("qfrc_actuator", env.data.qfrc_actuator)
        self._append("qfrc_constraint", env.data.qfrc_constraint)
        self._append("sensordata", env.data.sensordata)
        self._append("mujoco_state", integration_state)
        self._append("target_rotvec", np.zeros(3) if target_rotvec is None else target_rotvec)
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
                if image_capture_time_s is None
                or not np.isfinite(image_capture_time_s)
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
            # A rigid floating gripper has no actuators, so ctrl and
            # actuator_force legitimately have shape (T, 0). Zarr chunk
            # dimensions must still be positive.
            chunks = (min(256, len(value)),) + tuple(
                max(1, int(dimension)) for dimension in value.shape[1:]
            )
        group.array(
            name=key,
            data=value,
            chunks=chunks,
            compressor=compressor,
            overwrite=True,
        )

    def _update_meta(self) -> None:
        names = self.episode_names
        rgb_lengths = []
        robot_lengths = []
        wrench_lengths = []
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
        termination_reason: str,
        final_book_angle_deg: float,
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
            arrays["rgb_time_stamps_0"] = np.asarray(
                self._image_timestamps_ms, dtype=np.float64
            )
        else:
            width, height = self.image_size
            arrays["rgb_0"] = np.zeros((1, height, width, 3), dtype=np.uint8)
            arrays["rgb_time_stamps_0"] = np.zeros(1, dtype=np.float64)

        filtered = _causal_moving_average(
            arrays["wrench_0"],
            self.wrench_filter_samples,
        )
        arrays["wrench_filtered_0"] = filtered
        virtual_target, stiffness = adaptive_compliance_labels(
            arrays["ts_pose_command_0"],
            filtered,
            **self.adaptive_config,
        )
        arrays["ts_pose_virtual_target_0"] = virtual_target
        arrays["stiffness_0"] = stiffness

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
                    "final_book_angle_deg": float(final_book_angle_deg),
                    "final_task_metric_name": (
                        "book_angle_deg"
                        if final_task_metric_name is None
                        else str(final_task_metric_name)
                    ),
                    "final_task_metric_value": float(
                        final_book_angle_deg
                        if final_task_metric_value is None
                        else final_task_metric_value
                    ),
                    "sample_count": count,
                    "rgb_sample_count": len(arrays["rgb_0"]),
                    "adaptive_compliance": self.adaptive_config,
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


def validate_pyrite_dataset(dataset_path: str | Path) -> dict[str, Any]:
    """Validate schema, array alignment, rate, pose quaternions, and metadata."""
    zarr, _ = _zarr_modules()
    path = Path(dataset_path).expanduser().resolve()
    root = zarr.open(str(path), mode="r")
    if root.attrs.get("schema_name") != SCHEMA_NAME:
        raise ValueError(f"{path} is not a {SCHEMA_NAME} dataset")
    sample_hz = float(root.attrs["sample_hz"])
    period_ms = 1000.0 / sample_hz
    episode_names = sorted(
        root["data"].group_keys(),
        key=lambda key: int(key.rsplit("_", 1)[-1]),
    )
    if not episode_names:
        raise ValueError("dataset contains no episodes")
    lengths = []
    rgb_lengths = []
    for name in episode_names:
        episode = root["data"][name]
        missing = [key for key in PYRITE_REQUIRED_KEYS if key not in episode]
        if missing:
            raise ValueError(f"{name} is missing {missing}")
        count = len(episode["ts_pose_fb_0"])
        lengths.append(count)
        robot_keys = (
            "ts_pose_fb_0",
            "ts_pose_command_0",
            "ts_pose_virtual_target_0",
            "stiffness_0",
            "robot_time_stamps_0",
        )
        wrench_keys = ("wrench_0", "wrench_time_stamps_0")
        for key in robot_keys + wrench_keys:
            if len(episode[key]) != count:
                raise ValueError(
                    f"{name}/{key} has {len(episode[key])} rows, expected {count}"
                )
        rgb_count = len(episode["rgb_0"])
        rgb_lengths.append(rgb_count)
        if len(episode["rgb_time_stamps_0"]) != rgb_count:
            raise ValueError(
                f"{name}/rgb_time_stamps_0 has "
                f"{len(episode['rgb_time_stamps_0'])} rows, expected {rgb_count}"
            )
        times = np.asarray(episode["robot_time_stamps_0"])
        if count > 1 and not np.allclose(np.diff(times), period_ms, atol=1e-6):
            raise ValueError(f"{name} is not sampled at {sample_hz:g} Hz")
        rgb_times = np.asarray(episode["rgb_time_stamps_0"], dtype=float)
        if not np.all(np.isfinite(rgb_times)):
            raise ValueError(f"{name}/rgb_time_stamps_0 contains non-finite values")
        if rgb_count > 1 and np.any(np.diff(rgb_times) <= 0.0):
            raise ValueError(f"{name}/rgb_time_stamps_0 is not strictly increasing")
        if count and rgb_count and (
            rgb_times[0] < times[0] - period_ms
            or rgb_times[-1] > times[-1] + period_ms
        ):
            raise ValueError(
                f"{name}/rgb_time_stamps_0 [{rgb_times[0]:.3f}, "
                f"{rgb_times[-1]:.3f}] ms does not share the episode-relative "
                f"origin of robot timestamps [{times[0]:.3f}, "
                f"{times[-1]:.3f}] ms"
            )
        for key in ("ts_pose_fb_0", "ts_pose_command_0", "ts_pose_virtual_target_0"):
            quat_norm = np.linalg.norm(np.asarray(episode[key])[:, 3:], axis=1)
            if not np.allclose(quat_norm, 1.0, atol=1e-5):
                raise ValueError(f"{name}/{key} contains non-unit quaternions")

    meta_lengths = np.asarray(root["meta"]["episode_rgb0_len"])
    if not np.array_equal(meta_lengths, np.asarray(rgb_lengths)):
        raise ValueError(
            f"meta/episode_rgb0_len {meta_lengths.tolist()} does not match {rgb_lengths}"
        )
    robot_meta_lengths = np.asarray(root["meta"]["episode_robot0_len"])
    if not np.array_equal(robot_meta_lengths, np.asarray(lengths)):
        raise ValueError(
            f"meta/episode_robot0_len {robot_meta_lengths.tolist()} does not match {lengths}"
        )
    return {
        "path": str(path),
        "schema_version": int(root.attrs["schema_version"]),
        "sample_hz": sample_hz,
        "episodes": len(episode_names),
        "samples": int(sum(lengths)),
        "episode_lengths": lengths,
        "rgb_episode_lengths": rgb_lengths,
        "keys": sorted(root["data"][episode_names[0]].array_keys()),
    }
