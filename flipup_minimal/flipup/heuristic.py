from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from spatialmath import SE3, SO3
from spatialmath.base import q2r

from .environment import FlipUpEnv, matrix_to_pose7
from .physical_properties import (
    DEFAULT_PHYSICAL_PROPERTIES,
    DEFAULT_PHYSICAL_PROPERTY_RANGES,
    PhysicalProperties,
    PhysicalPropertyRanges,
    sample_physical_properties,
)
from .trajectory import MotionPlan, StaticTrajectory, task_space_trajectory

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FlipUpResult:
    seed: int
    success: bool
    final_angle_deg: float
    simulated_seconds: float
    wall_seconds: float
    viewer_closed: bool
    physical_properties: PhysicalProperties


def _tool_orientation(tool_position: FloatArray) -> SO3:
    robot_base_xy = np.array([-0.3, 0.0], dtype=np.float64)
    delta = tool_position[:2] - robot_base_xy
    if abs(delta[0]) <= 1e-5:
        raise ValueError("Tool target is too close to the robot base x-coordinate")
    yaw_deg = np.degrees(np.arctan(delta[1] / delta[0]))
    return SO3.RPY(0.0, -30.0, yaw_deg, unit="deg")


def _success_angle(book_pose: FloatArray) -> float:
    world_z = np.array([0.0, 0.0, 1.0])
    book_x = q2r(book_pose[3:])[:, 0]
    cosine = float(np.clip(np.dot(book_x, world_z), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def run_flipup(
    *,
    seed: int = 0,
    show_viewer: bool = True,
    success_threshold_deg: float = 15.0,
    verbose: bool = True,
    physical_properties: PhysicalProperties | None = None,
    randomize_physics: bool = False,
    physical_property_ranges: PhysicalPropertyRanges = (
        DEFAULT_PHYSICAL_PROPERTY_RANGES
    ),
) -> FlipUpResult:
    """Run one scripted FlipUp episode.

    Physical randomization is reproducible from ``seed`` and uses a separate
    random stream from scene-pose randomization.
    """
    if physical_properties is not None and randomize_physics:
        raise ValueError(
            "Pass either physical_properties or randomize_physics=True, not both"
        )
    if physical_properties is None:
        if randomize_physics:
            physical_properties = sample_physical_properties(
                seed,
                physical_property_ranges,
            )
        else:
            physical_properties = DEFAULT_PHYSICAL_PROPERTIES

    wall_start = time.monotonic()
    rng = np.random.RandomState(seed)

    bookend_transform = SE3.Rt(
        SO3.RPY(
            90.0,
            0.0,
            180.0 + rng.uniform(-10.0, 10.0),
            unit="deg",
        ),
        [
            0.3 + rng.uniform(0.0, 0.2),
            rng.uniform(-0.2, 0.2),
            0.2,
        ],
    )
    book_relative_transform = SE3.Rt(
        SO3.RPY(90.0, rng.uniform(0.0, 0.0), 0.0, unit="deg"),
        [
            0.015 + rng.uniform(0.0, 0.0),
            0.035,
            0.03 + rng.uniform(-0.02, 0.02),
        ],
    )
    book_transform = bookend_transform * book_relative_transform

    # Keep the original 1 cm trajectory clearance while adapting every
    # shape-dependent contact point to the sampled dimensions.
    book_length = physical_properties.length_m + 0.01
    upper_contact_height = physical_properties.thickness_m * 0.30
    lower_contact_height = physical_properties.thickness_m * 0.70
    thickness_offset = physical_properties.thickness_m * 0.28
    unit_x = np.array([1.0, 0.0, 0.0])
    unit_y = np.array([0.0, 1.0, 0.0])

    book_tip_contact = np.array(
        [
            physical_properties.length_m,
            physical_properties.width_m / 2.0,
            upper_contact_height,
        ]
    )
    bookend_tip_contact = np.asarray(
        (book_relative_transform * book_tip_contact).reshape(3),
        dtype=np.float64,
    )
    bookend_prepare = bookend_tip_contact + 0.05 * unit_x
    bookend_rotation_start = np.array(
        [
            thickness_offset + book_length,
            bookend_tip_contact[1],
            bookend_tip_contact[2],
        ]
    )

    rotation_waypoints = [bookend_rotation_start]
    for waypoint_id in range(21):
        angle = waypoint_id / 20.0 * np.pi / 2.0
        point = (
            bookend_rotation_start
            + (
                upper_contact_height * np.sin(angle)
                + book_length * np.cos(angle)
                - book_length
            )
            * unit_x
            + (
                lower_contact_height * np.cos(angle)
                + book_length * np.sin(angle)
                - lower_contact_height
            )
            * unit_y
        )
        rotation_waypoints.append(point)

    def target_pose(bookend_point: FloatArray) -> SE3:
        world_position = np.asarray(
            (bookend_transform * bookend_point).reshape(3),
            dtype=np.float64,
        )
        return SE3.Rt(_tool_orientation(world_position), world_position)

    flip_waypoints = [target_pose(point) for point in rotation_waypoints]
    prepare_pose = target_pose(bookend_prepare)
    engage_pose = flip_waypoints[0]

    if verbose:
        print(f"Creating FlipUp scene for seed {seed}...")
        print(f"Physics: {physical_properties.summary()}")

    viewer_closed = False
    with FlipUpEnv(
        np.asarray(bookend_transform.data[0], dtype=np.float64),
        np.asarray(book_transform.data[0], dtype=np.float64),
        show_viewer=show_viewer,
        physical_properties=physical_properties,
    ) as environment:
        tool_pose = environment.get_tool_pose()
        initial_pose = SE3.Rt(
            SO3(q2r(tool_pose[3:]), check=False),
            tool_pose[:3],
            check=False,
        )

        translation_velocity = np.full(3, 0.1)
        translation_acceleration = np.full(3, 5.0)
        rotation_velocity = np.full(3, 0.1)
        rotation_acceleration = np.full(3, 5.0)

        approach = task_space_trajectory(
            np.linspace(0.0, 1.0, 2),
            [initial_pose, prepare_pose],
            translation_velocity * 3.0,
            translation_acceleration * 3.0,
            rotation_velocity * 3.0,
            rotation_acceleration * 3.0,
        )
        engage = task_space_trajectory(
            np.linspace(0.0, 1.0, 2),
            [prepare_pose, engage_pose],
            translation_velocity * 0.5,
            translation_acceleration * 0.5,
            rotation_velocity * 0.5,
            rotation_acceleration * 0.5,
        )
        flip = task_space_trajectory(
            np.linspace(0.0, 5.0, len(flip_waypoints)),
            flip_waypoints,
            translation_velocity,
            translation_acceleration,
            rotation_velocity,
            rotation_acceleration,
        )
        plan = MotionPlan(
            [
                approach,
                engage,
                StaticTrajectory(flip(0.0), 0.5),
                flip,
                StaticTrajectory(flip(flip.duration), 1.5),
            ]
        )

        if verbose:
            print(f"Executing {plan.duration:.3f} s scripted trajectory...")

        timestep_index = 0
        termination_timestep = 6000
        while environment.current_time < plan.duration:
            target_transform = plan(environment.current_time)
            if not environment.step_task_space(matrix_to_pose7(target_transform)):
                viewer_closed = True
                break
            timestep_index += 1
            if timestep_index > termination_timestep:
                break

        final_angle_deg = _success_angle(environment.get_book_pose())
        simulated_seconds = environment.current_time

    result = FlipUpResult(
        seed=seed,
        success=final_angle_deg < success_threshold_deg,
        final_angle_deg=final_angle_deg,
        simulated_seconds=simulated_seconds,
        wall_seconds=time.monotonic() - wall_start,
        viewer_closed=viewer_closed,
        physical_properties=physical_properties,
    )

    if verbose:
        status = "SUCCESS" if result.success else "FAILURE"
        print(
            f"{status}: final angle={result.final_angle_deg:.2f} deg, "
            f"simulated={result.simulated_seconds:.3f} s, "
            f"wall={result.wall_seconds:.2f} s"
        )
    return result
