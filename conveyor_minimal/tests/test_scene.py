from __future__ import annotations

import numpy as np
import pytest

from conveyor.properties import DEFAULT_CUBE_PROPERTIES
from conveyor.scene import DEFAULT_LAYOUT, ConveyorLayout


def test_default_layout_matches_source_scene_config():
    layout = DEFAULT_LAYOUT
    assert layout.conveyor_center_xyz == (0.20, 0.24, 0.22)
    assert layout.conveyor_half_extents_xyz == (0.18, 0.60, 0.01)
    assert layout.conveyor_top_z == pytest.approx(0.23)
    assert layout.conveyor_start_y == pytest.approx(-0.36)
    assert layout.conveyor_end_y == pytest.approx(0.84)
    assert layout.target_bin_center_xyz == (0.50, 0.50, 0.24)


def test_shifted_moves_belt_bin_and_spawn_ranges_together():
    shifted = DEFAULT_LAYOUT.shifted(np.array([0.02, -0.01]))
    assert shifted.conveyor_center_xyz[0] == pytest.approx(0.22)
    assert shifted.conveyor_center_xyz[1] == pytest.approx(0.23)
    assert shifted.target_bin_center_xyz[0] == pytest.approx(0.52)
    assert shifted.target_bin_center_xyz[1] == pytest.approx(0.49)
    assert shifted.conveyor_spawn_x_range == pytest.approx(
        tuple(value + 0.02 for value in DEFAULT_LAYOUT.conveyor_spawn_x_range)
    )
    assert shifted.conveyor_spawn_y_range == pytest.approx(
        tuple(value - 0.01 for value in DEFAULT_LAYOUT.conveyor_spawn_y_range)
    )
    # The belt's own dimensions and the cube's belt-relative spawn band do not
    # change, so the task is the same task at a jittered position.
    assert shifted.conveyor_half_extents_xyz == DEFAULT_LAYOUT.conveyor_half_extents_xyz
    assert shifted.conveyor_end_y - shifted.conveyor_spawn_y_range[0] == pytest.approx(
        DEFAULT_LAYOUT.conveyor_end_y - DEFAULT_LAYOUT.conveyor_spawn_y_range[0]
    )


def test_shifted_leaves_the_original_untouched():
    before = DEFAULT_LAYOUT.conveyor_center_xyz
    DEFAULT_LAYOUT.shifted(np.array([0.05, 0.05]))
    assert DEFAULT_LAYOUT.conveyor_center_xyz == before


def test_on_conveyor_and_in_bin_predicates():
    layout = DEFAULT_LAYOUT
    assert layout.is_on_conveyor_xy(np.array([0.20, 0.0, 0.26]))
    assert not layout.is_on_conveyor_xy(np.array([0.60, 0.0, 0.26]))
    assert not layout.is_on_conveyor_xy(np.array([0.20, 1.0, 0.26]))

    bin_center = np.array(layout.target_bin_center_xyz)
    assert layout.is_in_target_bin(bin_center)
    assert layout.is_in_target_bin_xy(bin_center + np.array([0.0, 0.0, 1.0]))
    assert not layout.is_in_target_bin(bin_center + np.array([0.0, 0.0, 1.0]))
    assert layout.is_in_target_bin(bin_center + np.array([0.0, 0.0, 1.0]), ignore_height=True)


def test_spawn_pose_lands_on_the_belt_start_within_the_requested_band():
    rng = np.random.default_rng(0)
    layout = DEFAULT_LAYOUT
    properties = DEFAULT_CUBE_PROPERTIES
    for _ in range(200):
        pose = layout.sample_spawn_pose(
            properties.radius_m, properties.half_extent_m, rng
        )
        assert layout.is_on_conveyor_xy(pose[:3])
        assert (
            layout.conveyor_spawn_x_range[0] - 1e-9
            <= pose[0]
            <= layout.conveyor_spawn_x_range[1] + 1e-9
        )
        assert pose[2] == pytest.approx(
            layout.conveyor_top_z
            + properties.half_extent_m
            + layout.conveyor_height_tolerance
        )
        # Yaw only: the cube spawns flat.
        assert pose[4] == 0.0 and pose[5] == 0.0
        assert np.linalg.norm(pose[3:]) == pytest.approx(1.0)


def test_spawn_pose_clamps_instead_of_failing_for_an_oversized_object():
    layout = DEFAULT_LAYOUT
    rng = np.random.default_rng(0)
    # The source spawn y range is a single point 5.6 cm behind the start edge, so
    # a cube this size does not fit behind it. It should be nudged forward rather
    # than rejected.
    pose = layout.sample_spawn_pose(0.08, 0.03, rng)
    assert layout.is_on_conveyor_xy(pose[:3])
    assert pose[1] > layout.conveyor_spawn_y_range[1]


def test_spawn_pose_rejects_an_object_too_large_for_the_belt():
    with pytest.raises(ValueError):
        DEFAULT_LAYOUT.sample_spawn_pose(0.5, 0.5, np.random.default_rng(0))


def test_layout_validation():
    with pytest.raises(ValueError):
        ConveyorLayout(conveyor_half_extents_xyz=(0.18, 0.0, 0.01))
    with pytest.raises(ValueError):
        ConveyorLayout(conveyor_border_clearance=-0.01)


def test_belt_x_bounds_shrink_with_object_size():
    narrow = DEFAULT_LAYOUT.belt_x_bounds(0.0)
    wide = DEFAULT_LAYOUT.belt_x_bounds(0.05)
    assert wide[0] > narrow[0]
    assert wide[1] < narrow[1]
