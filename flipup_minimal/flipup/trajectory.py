from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint
from spatialmath import SE3, SO3

FloatArray = npt.NDArray[np.float64]

ta.setup_logging("WARNING")


class Motion:
    def __call__(self, time_point: float) -> FloatArray:
        raise NotImplementedError

    @property
    def duration(self) -> float:
        raise NotImplementedError


class SE3Trajectory(Motion):
    def __init__(self, joint_trajectory: object) -> None:
        self._joint_trajectory = joint_trajectory

    def __call__(self, time_point: float) -> FloatArray:
        waypoint = self._joint_trajectory(time_point)
        pose = SE3.Rt(
            SO3.RPY(waypoint[3], waypoint[4], waypoint[5]),
            waypoint[:3],
        )
        return np.asarray(pose.data[0], dtype=np.float64)

    @property
    def duration(self) -> float:
        return float(self._joint_trajectory.duration)


class StaticTrajectory(Motion):
    def __init__(self, transform: FloatArray, duration: float) -> None:
        self._transform = np.asarray(transform, dtype=np.float64)
        self._duration = float(duration)

    def __call__(self, time_point: float) -> FloatArray:
        del time_point
        return self._transform

    @property
    def duration(self) -> float:
        return self._duration


class MotionPlan(Motion):
    def __init__(self, motions: Sequence[Motion]) -> None:
        if not motions:
            raise ValueError("A motion plan requires at least one motion")
        self._motions = tuple(motions)
        self._durations = tuple(motion.duration for motion in motions)

    @property
    def duration(self) -> float:
        return float(sum(self._durations))

    def __call__(self, time_point: float) -> FloatArray:
        elapsed = 0.0
        for motion, duration in zip(self._motions, self._durations):
            if elapsed + duration >= time_point:
                return motion(time_point - elapsed)
            elapsed += duration
        raise ValueError(
            f"Query time {time_point} exceeds plan duration {self.duration}"
        )


def task_space_trajectory(
    time_stamps: FloatArray,
    waypoints: Sequence[SE3],
    translation_velocity_limits: FloatArray,
    translation_acceleration_limits: FloatArray,
    rotation_velocity_limits: FloatArray,
    rotation_acceleration_limits: FloatArray,
) -> SE3Trajectory:
    rpy_waypoints = np.array(
        [
            [
                pose.x,
                pose.y,
                pose.z,
                pose.rpy()[0],
                pose.rpy()[1],
                pose.rpy()[2],
            ]
            for pose in waypoints
        ],
        dtype=np.float64,
    )
    velocity_limits = np.concatenate(
        [translation_velocity_limits, rotation_velocity_limits]
    )
    acceleration_limits = np.concatenate(
        [translation_acceleration_limits, rotation_acceleration_limits]
    )

    path = ta.SplineInterpolator(time_stamps, rpy_waypoints)
    constraints = [
        constraint.JointVelocityConstraint(velocity_limits),
        constraint.JointAccelerationConstraint(acceleration_limits),
    ]
    instance = algo.TOPPRA(
        constraints,
        path,
        parametrizer="ParametrizeConstAccel",
    )
    trajectory = instance.compute_trajectory()
    if trajectory is None:
        raise RuntimeError("TOPPRA could not parameterize the requested trajectory")
    return SE3Trajectory(trajectory)
