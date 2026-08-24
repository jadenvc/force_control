from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .environment import ConveyorEnv, grasp_pose7
from .judge import ConveyorJudge
from .properties import (
    DEFAULT_BELT_SPEED_RANGE,
    DEFAULT_CUBE_PROPERTIES,
    CubeProperties,
    CubePropertyRanges,
    DEFAULT_CUBE_PROPERTY_RANGES,
    ValueRange,
    sample_cube_properties,
)
from .scene import DEFAULT_LAYOUT, ConveyorLayout

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ConveyorResult:
    seed: int
    episode_index: int
    success: bool
    termination_reason: str
    belt_speed_m_per_s: float
    estimated_belt_speed_m_per_s: float
    picked_up: bool
    simulated_seconds: float
    wall_seconds: float
    viewer_closed: bool
    cube_properties: CubeProperties
    final_phase: str


@dataclass
class ConveyorHeuristic:
    """Closed-loop scripted pick-and-place for the moving belt.

    The state machine is the source task's ``ConveyorPickPlaceAgent`` reduced to
    the parts the task needs: estimate the belt speed from the cube's observed
    motion, hover ahead of the cube, descend onto it, close, lift, carry to the
    bin, release, retreat.

    Nothing here reads ``env.conveyor_speed_m_per_s``. The belt speed is
    estimated from successive cube positions exactly as an agent with only
    observations would have to, which is also what makes it a usable baseline
    when the speed is randomized on every reset.
    """

    env: ConveyorEnv

    # The state machine and the belt-speed estimate run at this rate; the
    # commanded target is slewed toward the current goal at every simulator step.
    agent_update_hz: float = 20.0

    # Motion limits for the commanded target.
    position_speed_m_per_s: float = 0.74
    gripper_close_speed_m_per_s: float = 0.30
    gripper_open_speed_m_per_s: float = 0.13
    lift_position_speed_scale: float = 0.55

    # Heights, all relative to the belt surface or the bin floor.
    hover_height_over_conveyor_m: float = 0.14
    grasp_height_over_object_m: float = -0.034
    lift_height_over_conveyor_m: float = 0.20
    bin_travel_height_over_center_m: float = 0.20
    bin_release_height_over_center_m: float = 0.08
    retreat_height_over_conveyor_m: float = 0.18

    open_gripper_width_m: float = 0.10
    close_gripper_width_m: float = 0.02

    # Tracking and grasp timing. The alignment tolerances have to be wider than
    # the impedance controller's tracking lag, which is roughly
    # (drag force)/task_space_kp and reaches ~10 mm while following the belt at
    # 0.3 m/s. Tighter values leave the tool chasing the cube down the whole belt
    # without ever declaring itself aligned. They are still far inside what the
    # compliant fin-ray fingers can grasp: a 10 cm opening onto a 5 cm cube.
    observation_wait_s: float = 0.25
    speed_estimate_ema_alpha: float = 0.35
    # The lead also absorbs that lag, so the tool arrives with the cube rather
    # than behind it.
    lead_time_s: float = 0.09
    lead_time_speed_scale_s_per_m: float = 0.06
    lead_time_max_s: float = 0.16
    xy_align_tolerance_m: float = 0.030
    z_align_tolerance_m: float = 0.015
    descend_trigger_y_error_m: float = 0.10
    grasp_close_time_s: float = 0.35
    pick_success_height_margin_m: float = 0.08
    max_lift_time_s: float = 2.0
    bin_align_tolerance_m: float = 0.02
    release_wait_time_s: float = 0.4
    max_release_wait_time_s: float = 1.5
    gripper_open_tolerance_m: float = 0.005
    retreat_tolerance_m: float = 0.03
    belt_x_goal_margin_m: float = 0.03
    max_grasp_depth_below_belt_m: float = 0.012

    phase: str = field(default="waiting", init=False)
    phase_start_time_s: float = field(default=0.0, init=False)
    estimated_speed_m_per_s: float = field(default=0.0, init=False)
    estimated_speed_at_grasp_m_per_s: float = field(default=0.0, init=False)
    target_pose: FloatArray = field(init=False, repr=False)
    gripper_width_cmd: float = field(default=0.10, init=False)

    def __post_init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------ helpers
    def reset(self) -> None:
        self.phase = "waiting"
        self.phase_start_time_s = self.env.current_time
        self.estimated_speed_m_per_s = 0.0
        self.estimated_speed_at_grasp_m_per_s = 0.0
        self._last_object_y: float | None = None
        self._last_observation_time_s: float | None = None
        self._next_update_time_s: float | None = None
        self.target_pose = self.env.tool_pose.copy()
        self.gripper_width_cmd = self.open_gripper_width_m
        self._goal_position = self.target_pose[:3].copy()
        self._goal_width = self.open_gripper_width_m

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.phase_start_time_s = self.env.current_time

    @property
    def phase_elapsed_s(self) -> float:
        return self.env.current_time - self.phase_start_time_s

    def _update_speed_estimate(self) -> None:
        """One EMA-filtered finite difference of the cube's observed y."""
        now = self.env.current_time
        object_y = float(self.env.object_pose[1])
        if self._last_observation_time_s is not None and self._last_object_y is not None:
            interval = now - self._last_observation_time_s
            if interval > 0.0:
                measured = (object_y - self._last_object_y) / interval
                if self.estimated_speed_m_per_s == 0.0:
                    self.estimated_speed_m_per_s = max(0.0, measured)
                else:
                    alpha = self.speed_estimate_ema_alpha
                    self.estimated_speed_m_per_s = float(
                        np.clip(
                            (1.0 - alpha) * self.estimated_speed_m_per_s
                            + alpha * measured,
                            0.0,
                            None,
                        )
                    )
        self._last_object_y = object_y
        self._last_observation_time_s = now

    def _lead_y_m(self) -> float:
        speed = self.estimated_speed_m_per_s
        lead_time = min(
            self.lead_time_s + self.lead_time_speed_scale_s_per_m * speed,
            self.lead_time_max_s,
        )
        return speed * lead_time

    def _belt_x_goal(self, object_x: float) -> float:
        """Cube x, clamped to the belt band the arm can work in comfortably.

        Near the rails the arm reaches its own limits and the impedance
        controller stalls, so a cube nudged against a rail must not drag the tool
        out there with it.
        """
        low, high = self.env.layout.belt_x_bounds(
            self.env.cube_properties.radius_m + self.belt_x_goal_margin_m
        )
        return float(np.clip(object_x, low, high))

    def _grasp_z_goal(self, object_z: float) -> float:
        """Grasp height, kept from driving the fingers into the belt surface.

        The tool site sits just past the fingertips, so the nominal grasp height
        is a little below the cube's centre; the clamp only guards against a
        tilted or mis-read cube pose.
        """
        floor = self.env.layout.conveyor_top_z - self.max_grasp_depth_below_belt_m
        return float(max(object_z + self.grasp_height_over_object_m, floor))

    def _intercept_y_m(self) -> float:
        """Belt y where the tool should meet the cube.

        Aim at where the cube will be once the tool has closed the distance,
        clamped to stay clear of the belt's far end.
        """
        layout = self.env.layout
        object_y = float(self.env.object_pose[1])
        travel = float(
            np.linalg.norm(self.env.tool_pose[:2] - self.env.object_pose[:2])
        )
        time_to_reach = travel / max(self.position_speed_m_per_s, 1e-6)
        intercept = object_y + self.estimated_speed_m_per_s * time_to_reach
        return float(
            min(intercept, layout.conveyor_end_y - 3.0 * self.xy_align_tolerance_m)
        )

    # ------------------------------------------------------------- state machine
    def _evaluate_phase(self) -> tuple[FloatArray, float]:
        """Goal tool position and gripper width for the current phase."""
        env = self.env
        layout = env.layout
        object_pose = env.object_pose
        tool_pose = env.tool_pose
        lead_y = self._lead_y_m()
        grasp_x = self._belt_x_goal(object_pose[0])
        grasp_xy = np.array([grasp_x, object_pose[1] + lead_y])
        hover_z = layout.conveyor_top_z + self.hover_height_over_conveyor_m
        home = env.home_tool_pose[:3]

        if self.phase == "waiting":
            if (
                self.phase_elapsed_s >= self.observation_wait_s
                and self.estimated_speed_m_per_s > 0.0
                and env.object_on_conveyor
            ):
                self._set_phase("tracking")
            return home, self.open_gripper_width_m

        if self.phase == "tracking":
            intercept_y = self._intercept_y_m()
            goal = np.array([grasp_x, intercept_y, hover_z])
            aligned = (
                float(np.linalg.norm(tool_pose[:2] - goal[:2]))
                <= self.xy_align_tolerance_m
            )
            approaching = (
                object_pose[1] >= intercept_y - self.descend_trigger_y_error_m
            )
            if aligned and approaching:
                # Record the estimate here: from now on the cube is being
                # touched, so its motion stops reporting the belt alone.
                self.estimated_speed_at_grasp_m_per_s = self.estimated_speed_m_per_s
                self._set_phase("descending")
            return goal, self.open_gripper_width_m

        if self.phase == "descending":
            grasp_z = self._grasp_z_goal(object_pose[2])
            goal = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
            if (
                float(np.linalg.norm(tool_pose[:2] - goal[:2]))
                <= self.xy_align_tolerance_m
                and abs(tool_pose[2] - grasp_z) <= self.z_align_tolerance_m
            ):
                self._set_phase("closing")
            return goal, self.open_gripper_width_m

        if self.phase == "closing":
            grasp_z = self._grasp_z_goal(object_pose[2])
            goal = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
            if self.phase_elapsed_s >= self.grasp_close_time_s:
                self._set_phase("lifting")
            return goal, self.close_gripper_width_m

        if self.phase == "lifting":
            goal = np.array(
                [
                    grasp_xy[0],
                    grasp_xy[1],
                    layout.conveyor_top_z + self.lift_height_over_conveyor_m,
                ]
            )
            lifted = (
                object_pose[2]
                > layout.conveyor_top_z + self.pick_success_height_margin_m
            )
            if lifted:
                self._set_phase("moving_to_bin")
            elif self.phase_elapsed_s >= self.max_lift_time_s:
                # The grasp failed. Re-open and go back to tracking the cube.
                self._set_phase("waiting")
                return home, self.open_gripper_width_m
            return goal, self.close_gripper_width_m

        if self.phase == "moving_to_bin":
            goal = np.array(
                [
                    layout.target_bin_center_xyz[0],
                    layout.target_bin_center_xyz[1],
                    layout.target_bin_center_xyz[2]
                    + self.bin_travel_height_over_center_m,
                ]
            )
            if float(np.linalg.norm(tool_pose[:3] - goal)) <= self.bin_align_tolerance_m:
                self._set_phase("lowering_to_place")
            return goal, self.close_gripper_width_m

        if self.phase == "lowering_to_place":
            goal = np.array(
                [
                    layout.target_bin_center_xyz[0],
                    layout.target_bin_center_xyz[1],
                    layout.target_bin_center_xyz[2]
                    + self.bin_release_height_over_center_m,
                ]
            )
            if abs(tool_pose[2] - goal[2]) <= self.z_align_tolerance_m:
                self._set_phase("releasing")
            return goal, self.close_gripper_width_m

        if self.phase == "releasing":
            goal = np.array(
                [
                    layout.target_bin_center_xyz[0],
                    layout.target_bin_center_xyz[1],
                    layout.target_bin_center_xyz[2]
                    + self.bin_release_height_over_center_m,
                ]
            )
            # Wait for the fingers to have actually opened, not just for a fixed
            # time: opening 8 cm at the WSG50's slew rate takes longer than the
            # dwell, and retreating early drags the cube back out of the bin.
            opened = (
                env.gripper_width
                >= self.open_gripper_width_m - self.gripper_open_tolerance_m
            )
            if (
                opened and self.phase_elapsed_s >= self.release_wait_time_s
            ) or self.phase_elapsed_s >= self.max_release_wait_time_s:
                self._set_phase("retreating")
            return goal, self.open_gripper_width_m

        if self.phase == "retreating":
            goal = np.array(
                [
                    home[0],
                    home[1],
                    max(
                        home[2],
                        layout.conveyor_top_z + self.retreat_height_over_conveyor_m,
                    ),
                ]
            )
            if float(np.linalg.norm(tool_pose[:3] - goal)) <= self.retreat_tolerance_m:
                self._set_phase("waiting")
            return goal, self.open_gripper_width_m

        raise ValueError(f"Unknown phase {self.phase!r}")

    def step(self) -> tuple[FloatArray, float]:
        """Advance the state machine and return the slewed command for one step."""
        now = self.env.current_time
        if self._next_update_time_s is None or now >= self._next_update_time_s:
            self._update_speed_estimate()
            self._goal_position, self._goal_width = self._evaluate_phase()
            self._next_update_time_s = now + 1.0 / self.agent_update_hz

        goal_position, goal_width = self._goal_position, self._goal_width
        timestep = self.env.timestep

        speed = self.position_speed_m_per_s
        if self.phase == "lifting":
            speed *= self.lift_position_speed_scale
        delta = np.asarray(goal_position, dtype=np.float64) - self.target_pose[:3]
        distance = float(np.linalg.norm(delta))
        max_step = speed * timestep
        if distance <= max_step:
            self.target_pose[:3] = goal_position
        else:
            self.target_pose[:3] = self.target_pose[:3] + delta * (max_step / distance)
        self.target_pose = grasp_pose7(self.target_pose[:3])

        width_error = goal_width - self.gripper_width_cmd
        width_speed = (
            self.gripper_close_speed_m_per_s
            if width_error < 0.0
            else self.gripper_open_speed_m_per_s
        )
        width_step = width_speed * timestep
        if abs(width_error) <= width_step:
            self.gripper_width_cmd = goal_width
        else:
            self.gripper_width_cmd += np.sign(width_error) * width_step

        return self.target_pose, self.gripper_width_cmd


def run_conveyor(
    *,
    seed: int = 0,
    episode_index: int = 0,
    show_viewer: bool = True,
    verbose: bool = True,
    layout: ConveyorLayout = DEFAULT_LAYOUT,
    cube_properties: CubeProperties | None = None,
    randomize_cube: bool = False,
    cube_property_ranges: CubePropertyRanges = DEFAULT_CUBE_PROPERTY_RANGES,
    belt_speed_m_per_s: float | None = None,
    belt_speed_range: ValueRange = DEFAULT_BELT_SPEED_RANGE,
    randomize_belt_speed: bool = True,
    randomize_layout: bool = True,
    time_limit_s: float = 30.0,
) -> ConveyorResult:
    """Run one scripted conveyor pick-and-place episode.

    Cube randomization is reproducible from ``seed`` and uses a random stream
    separate from the belt speed, layout and spawn pose, so enabling it does not
    change the episode an existing seed produces.
    """
    if cube_properties is not None and randomize_cube:
        raise ValueError(
            "Pass either cube_properties or randomize_cube=True, not both"
        )
    if cube_properties is None:
        cube_properties = (
            sample_cube_properties(seed, cube_property_ranges)
            if randomize_cube
            else DEFAULT_CUBE_PROPERTIES
        )

    wall_start = time.monotonic()
    viewer_closed = False

    with ConveyorEnv(
        layout=layout,
        cube_properties=cube_properties,
        belt_speed_m_per_s=belt_speed_m_per_s,
        belt_speed_range=belt_speed_range,
        randomize_belt_speed=randomize_belt_speed,
        randomize_layout=randomize_layout,
        seed=seed,
        show_viewer=show_viewer,
    ) as env:
        env.reset(episode_index=episode_index)
        judge = ConveyorJudge(time_limit_s=time_limit_s)
        judge.reset()
        agent = ConveyorHeuristic(env)

        if verbose:
            print(
                f"Conveyor episode: seed={seed} index={episode_index} "
                f"belt={env.conveyor_speed_m_per_s:.3f} m/s "
                f"layout_offset=({env.layout_offset_xy[0]:+.3f}, "
                f"{env.layout_offset_xy[1]:+.3f}) m"
            )
            print(f"Cube: {cube_properties.summary()}")

        while not judge.done(env):
            target_pose, gripper_width = agent.step()
            if not env.step_task_space(target_pose, gripper_width):
                viewer_closed = True
                break
            judge.update(env)

        result = ConveyorResult(
            seed=seed,
            episode_index=episode_index,
            success=judge.success,
            termination_reason=judge.termination_reason,
            belt_speed_m_per_s=env.conveyor_speed_m_per_s,
            estimated_belt_speed_m_per_s=agent.estimated_speed_at_grasp_m_per_s,
            picked_up=judge.object_picked_up,
            simulated_seconds=env.current_time,
            wall_seconds=time.monotonic() - wall_start,
            viewer_closed=viewer_closed,
            cube_properties=cube_properties,
            final_phase=agent.phase,
        )

    if verbose:
        status = "SUCCESS" if result.success else "FAILURE"
        print(
            f"{status}: {result.termination_reason}, "
            f"belt={result.belt_speed_m_per_s:.3f} m/s "
            f"(estimated {result.estimated_belt_speed_m_per_s:.3f}), "
            f"phase={result.final_phase}, "
            f"simulated={result.simulated_seconds:.2f} s, "
            f"wall={result.wall_seconds:.2f} s"
        )
    return result
