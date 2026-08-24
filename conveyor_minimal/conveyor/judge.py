from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .environment import ConveyorEnv


@dataclass
class ConveyorJudge:
    """Success, miss and failure bookkeeping for one conveyor episode.

    The thresholds are the source task's
    ``ConveyorPickPlaceJudge`` values (``config/task/conveyor_pick_place.yaml``).
    An episode succeeds when the cube is first lifted clear of the belt with the
    tool next to it, and then lands inside the target bin.
    """

    lift_height_threshold: float = 0.12
    place_height_threshold: float = 0.04
    tcp_object_distance_threshold: float = 0.08
    object_missed_y_margin: float = 0.0
    object_fall_below_conveyor_margin: float = 0.01
    terminate_on_success: bool = True
    terminate_on_miss: bool = True
    terminate_on_fall: bool = True
    prolong_after_success_s: float = 0.5
    time_limit_s: float = 30.0

    object_picked_up: bool = field(default=False, init=False)
    object_placed_in_target_bin: bool = field(default=False, init=False)
    object_missed: bool = field(default=False, init=False)
    object_fell_to_ground: bool = field(default=False, init=False)
    time_limit_reached: bool = field(default=False, init=False)
    finished_timestamp: float | None = field(default=None, init=False)

    def reset(self) -> None:
        self.object_picked_up = False
        self.object_placed_in_target_bin = False
        self.object_missed = False
        self.object_fell_to_ground = False
        self.time_limit_reached = False
        self.finished_timestamp = None

    def update(self, env: ConveyorEnv) -> None:
        timestamp = env.current_time
        if timestamp > self.time_limit_s:
            self.time_limit_reached = True

        layout = env.layout
        object_pose = env.object_pose
        tool_pose = env.tool_pose

        if (
            not layout.is_on_conveyor_xy(object_pose[:3])
            and not layout.is_in_target_bin(object_pose[:3])
            and object_pose[2]
            <= layout.conveyor_top_z - self.object_fall_below_conveyor_margin
        ):
            self.object_fell_to_ground = True
            return

        if not self.object_picked_up:
            miss_y = layout.conveyor_end_y - max(0.0, self.object_missed_y_margin)
            if object_pose[1] >= miss_y:
                self.object_missed = True
                return
            tool_object_distance = float(
                np.linalg.norm(object_pose[:3] - tool_pose[:3])
            )
            if (
                object_pose[2] > layout.conveyor_top_z + self.lift_height_threshold
                and tool_object_distance < self.tcp_object_distance_threshold
            ):
                self.object_picked_up = True
            return

        if not self.object_placed_in_target_bin:
            landed = (
                object_pose[2]
                < layout.target_bin_center_xyz[2] + self.place_height_threshold
            )
            if layout.is_in_target_bin_xy(object_pose[:3]) and landed:
                self.object_placed_in_target_bin = True
                self.finished_timestamp = timestamp

    @property
    def success(self) -> bool:
        if self.time_limit_reached:
            return False
        return bool(self.object_picked_up and self.object_placed_in_target_bin)

    def done(self, env: ConveyorEnv) -> bool:
        if self.time_limit_reached:
            return True
        if self.terminate_on_miss and self.object_missed:
            return True
        if self.terminate_on_fall and self.object_fell_to_ground:
            return True
        if not self.terminate_on_success or self.finished_timestamp is None:
            return False
        if self.prolong_after_success_s <= 0.0:
            return True
        return env.current_time - self.finished_timestamp > self.prolong_after_success_s

    @property
    def termination_reason(self) -> str:
        if self.object_placed_in_target_bin:
            return "success"
        if self.object_missed:
            return "object_missed"
        if self.object_fell_to_ground:
            return "object_fell"
        if self.time_limit_reached:
            return "time_limit"
        return "running"

    def states(self) -> dict[str, bool]:
        return {
            "object_picked_up": self.object_picked_up,
            "object_placed_in_target_bin": self.object_placed_in_target_bin,
            "object_missed": self.object_missed,
            "object_fell_to_ground": self.object_fell_to_ground,
            "time_limit_reached": self.time_limit_reached,
        }
