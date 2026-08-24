from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from .properties import ValueRange

FloatArray = npt.NDArray[np.float64]

# Coordinate system, matching the source task:
#
#   ---> y     the belt carries the cube in +y
#   |
#   v x
#
# The robot stands at +x, the belt runs along y, and the target bin sits past
# the belt's far corner.


@dataclass(frozen=True)
class ConveyorLayout:
    """Nominal belt and bin geometry, plus the episode randomization ranges.

    Every value is the source task's
    ``config/task/env/scene/table_conveyor_pick_place.yaml`` value, so a policy
    or demonstration collected here sees the same workspace. The layout is
    immutable: :meth:`shifted` returns the per-episode jittered copy.
    """

    conveyor_center_xyz: tuple[float, float, float] = (0.20, 0.24, 0.22)
    conveyor_half_extents_xyz: tuple[float, float, float] = (0.18, 0.60, 0.01)
    # Spawn at the belt's start edge, with x free across most of the width.
    conveyor_spawn_x_range: tuple[float, float] = (0.12, 0.28)
    conveyor_spawn_y_range: tuple[float, float] = (-0.304, -0.304)
    conveyor_border_clearance: float = 0.02
    conveyor_height_tolerance: float = 0.002

    target_bin_center_xyz: tuple[float, float, float] = (0.50, 0.50, 0.24)
    target_bin_half_extents_xyz: tuple[float, float, float] = (0.09, 0.07, 0.06)
    target_bin_height_tolerance: float = 0.02

    cube_yaw_range: tuple[float, float] = (-0.25, 0.25)

    # Per-episode layout jitter. These are the ranges the source task's data
    # collection used (shell_scripts/collect_data_by_speed.sh); the target bin
    # follows the belt so the pick-to-place relation is preserved.
    layout_offset_x_range: ValueRange = ValueRange(-0.02, 0.02)
    layout_offset_y_range: ValueRange = ValueRange(-0.01, 0.01)

    def __post_init__(self) -> None:
        for name in (
            "conveyor_center_xyz",
            "conveyor_half_extents_xyz",
            "target_bin_center_xyz",
            "target_bin_half_extents_xyz",
        ):
            if len(getattr(self, name)) != 3:
                raise ValueError(f"{name} must have three components")
        if any(value <= 0.0 for value in self.conveyor_half_extents_xyz):
            raise ValueError("conveyor_half_extents_xyz must be positive")
        if any(value <= 0.0 for value in self.target_bin_half_extents_xyz):
            raise ValueError("target_bin_half_extents_xyz must be positive")
        if self.conveyor_border_clearance < 0.0:
            raise ValueError("conveyor_border_clearance cannot be negative")

    # --------------------------------------------------------------- geometry
    @property
    def conveyor_top_z(self) -> float:
        return float(self.conveyor_center_xyz[2] + self.conveyor_half_extents_xyz[2])

    @property
    def conveyor_start_y(self) -> float:
        return float(self.conveyor_center_xyz[1] - self.conveyor_half_extents_xyz[1])

    @property
    def conveyor_end_y(self) -> float:
        return float(self.conveyor_center_xyz[1] + self.conveyor_half_extents_xyz[1])

    @property
    def conveyor_length_m(self) -> float:
        return float(2.0 * self.conveyor_half_extents_xyz[1])

    def belt_x_bounds(self, object_radius: float) -> tuple[float, float]:
        """Belt x range that keeps an object of ``object_radius`` off the rails."""
        margin = float(object_radius) + self.conveyor_border_clearance
        return (
            float(self.conveyor_center_xyz[0] - self.conveyor_half_extents_xyz[0] + margin),
            float(self.conveyor_center_xyz[0] + self.conveyor_half_extents_xyz[0] - margin),
        )

    def is_on_conveyor_xy(self, position_xyz: FloatArray) -> bool:
        position_xyz = np.asarray(position_xyz, dtype=np.float64)
        return bool(
            abs(position_xyz[0] - self.conveyor_center_xyz[0])
            <= self.conveyor_half_extents_xyz[0]
            and abs(position_xyz[1] - self.conveyor_center_xyz[1])
            <= self.conveyor_half_extents_xyz[1]
        )

    def is_in_target_bin_xy(self, position_xyz: FloatArray) -> bool:
        position_xyz = np.asarray(position_xyz, dtype=np.float64)
        return bool(
            abs(position_xyz[0] - self.target_bin_center_xyz[0])
            <= self.target_bin_half_extents_xyz[0]
            and abs(position_xyz[1] - self.target_bin_center_xyz[1])
            <= self.target_bin_half_extents_xyz[1]
        )

    def is_in_target_bin(
        self,
        position_xyz: FloatArray,
        *,
        ignore_height: bool = False,
    ) -> bool:
        if not self.is_in_target_bin_xy(position_xyz):
            return False
        if ignore_height:
            return True
        position_xyz = np.asarray(position_xyz, dtype=np.float64)
        return bool(
            abs(position_xyz[2] - self.target_bin_center_xyz[2])
            <= self.target_bin_half_extents_xyz[2] + self.target_bin_height_tolerance
        )

    # ---------------------------------------------------------- randomization
    def shifted(self, offset_xy: FloatArray) -> "ConveyorLayout":
        """Return this layout with belt, bin and spawn ranges moved by ``offset_xy``.

        The target bin follows the belt, and the spawn ranges move with it, so a
        jittered episode keeps the same belt-relative task.
        """
        offset = np.asarray(offset_xy, dtype=np.float64).reshape(2)
        return replace(
            self,
            conveyor_center_xyz=(
                self.conveyor_center_xyz[0] + float(offset[0]),
                self.conveyor_center_xyz[1] + float(offset[1]),
                self.conveyor_center_xyz[2],
            ),
            target_bin_center_xyz=(
                self.target_bin_center_xyz[0] + float(offset[0]),
                self.target_bin_center_xyz[1] + float(offset[1]),
                self.target_bin_center_xyz[2],
            ),
            conveyor_spawn_x_range=(
                self.conveyor_spawn_x_range[0] + float(offset[0]),
                self.conveyor_spawn_x_range[1] + float(offset[0]),
            ),
            conveyor_spawn_y_range=(
                self.conveyor_spawn_y_range[0] + float(offset[1]),
                self.conveyor_spawn_y_range[1] + float(offset[1]),
            ),
        )

    def sample_layout_offset(self, rng: np.random.Generator) -> FloatArray:
        return np.array(
            [
                self.layout_offset_x_range.sample(rng),
                self.layout_offset_y_range.sample(rng),
            ],
            dtype=np.float64,
        )

    def sample_spawn_pose(
        self,
        object_radius: float,
        object_center_height: float,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Sample an xyz + wxyz spawn pose on the belt's start edge.

        The requested spawn ranges are clamped into the band that keeps an object
        of ``object_radius`` clear of the rails and the belt ends, rather than
        rejected. The source task's y range is a single point right at the start
        edge, which a randomized larger cube would otherwise not fit behind.
        """
        belt_x_low, belt_x_high = self.belt_x_bounds(object_radius)
        if belt_x_low > belt_x_high:
            raise ValueError(
                f"An object of radius {object_radius} does not fit across the belt"
            )
        margin = float(object_radius) + self.conveyor_border_clearance
        belt_y_low = self.conveyor_start_y + margin
        belt_y_high = self.conveyor_end_y - margin
        if belt_y_low > belt_y_high:
            raise ValueError(
                f"An object of radius {object_radius} does not fit along the belt"
            )

        x_low, x_high = (
            float(np.clip(value, belt_x_low, belt_x_high))
            for value in sorted(self.conveyor_spawn_x_range)
        )
        y_low, y_high = (
            float(np.clip(value, belt_y_low, belt_y_high))
            for value in sorted(self.conveyor_spawn_y_range)
        )

        position_x = float(rng.uniform(x_low, x_high))
        position_y = float(rng.uniform(y_low, y_high))
        position_z = (
            self.conveyor_top_z
            + float(object_center_height)
            + self.conveyor_height_tolerance
        )
        yaw = float(rng.uniform(*self.cube_yaw_range))

        return np.array(
            [
                position_x,
                position_y,
                position_z,
                math.cos(yaw / 2.0),
                0.0,
                0.0,
                math.sin(yaw / 2.0),
            ],
            dtype=np.float64,
        )


DEFAULT_LAYOUT = ConveyorLayout()
