from __future__ import annotations

import numpy as np
import pytest

from conveyor.environment import GRASP_QUAT_WXYZ, ConveyorEnv, grasp_pose7
from conveyor.properties import CubeProperties, ValueRange

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def env():
    environment = ConveyorEnv(show_viewer=False, seed=0)
    yield environment
    environment.close()


def _slew_to(environment, goal_position, *, seconds=4.0, speed=0.35, width=None):
    """Drive the tool to a pose the way every caller must: without step jumps.

    The task-space controller saturates the UR5e's actuators if handed a large
    step, so a commanded target is always slewed. Returns the final tool pose.
    """
    if width is None:
        width = environment.OPEN_GRIPPER_WIDTH_M
    goal = grasp_pose7(goal_position)
    target = environment.tool_pose.copy()
    for _ in range(int(seconds / environment.timestep)):
        delta = goal[:3] - target[:3]
        distance = float(np.linalg.norm(delta))
        step = speed * environment.timestep
        target[:3] = goal[:3] if distance <= step else target[:3] + delta * (step / distance)
        environment.step_task_space(grasp_pose7(target[:3]), width)
    return environment.tool_pose


def test_model_assembly(env):
    assert env.timestep == pytest.approx(0.001)
    assert env.model.actuator("ur5e/wsg50/gripper").id == env.gripper_actuator_id
    assert env.model.site("ur5e/wsg50/end_effector").id == env.tool_site_id
    assert env.model.geom("conveyor_belt_surface").id == env.belt_surface_geom_id
    assert len(env._object_geom_ids) >= 1
    # A freejoint for the cube plus six arm joints and two gripper drivers.
    assert env.model.nq == 7 + 6 + 2


def test_reset_places_the_tool_at_its_home_pose(env):
    env.reset(episode_index=0)
    pose = env.tool_pose
    home = env.home_tool_pose
    assert np.linalg.norm(pose[:3] - home[:3]) < 2e-3
    assert abs(abs(float(np.dot(pose[3:], home[3:]))) - 1.0) < 1e-4
    assert env.gripper_width == pytest.approx(env.OPEN_GRIPPER_WIDTH_M, abs=2e-3)


def test_reset_starts_the_cube_at_rest_on_the_belt(env):
    env.reset(episode_index=3)
    pose = env.object_pose
    assert env.layout.is_on_conveyor_xy(pose[:3])
    assert pose[1] < env.layout.conveyor_start_y + 0.10
    assert np.linalg.norm(env.object_velocity) < 0.02
    assert env.current_time == pytest.approx(0.0)


def test_reset_randomizes_the_belt_speed_and_is_reproducible_per_index(env):
    speeds = []
    for index in range(8):
        env.reset(episode_index=index)
        speeds.append(env.conveyor_speed_m_per_s)
    assert len(set(speeds)) == len(speeds), "every reset should draw a new speed"
    for speed in speeds:
        assert 0.01 <= speed <= 0.30

    repeated = []
    for index in range(8):
        env.reset(episode_index=index)
        repeated.append(env.conveyor_speed_m_per_s)
    assert repeated == speeds


def test_successive_resets_advance_the_episode_index(env):
    env.reset(episode_index=0)
    first = env.conveyor_speed_m_per_s
    env.reset()
    second = env.conveyor_speed_m_per_s
    assert first != second


def test_reset_randomizes_the_layout_within_the_configured_ranges(env):
    offsets = []
    for index in range(10):
        env.reset(episode_index=index)
        offsets.append(env.layout_offset_xy.copy())
        assert env.layout.conveyor_center_xyz[0] == pytest.approx(
            env.nominal_layout.conveyor_center_xyz[0] + env.layout_offset_xy[0]
        )
        assert env.layout.target_bin_center_xyz[1] == pytest.approx(
            env.nominal_layout.target_bin_center_xyz[1] + env.layout_offset_xy[1]
        )
    offsets = np.array(offsets)
    ranges = env.nominal_layout
    assert np.all(offsets[:, 0] >= ranges.layout_offset_x_range.minimum)
    assert np.all(offsets[:, 0] <= ranges.layout_offset_x_range.maximum)
    assert np.all(offsets[:, 1] >= ranges.layout_offset_y_range.minimum)
    assert np.all(offsets[:, 1] <= ranges.layout_offset_y_range.maximum)
    assert offsets.std(axis=0).min() > 0.0


def test_pinned_belt_speed_survives_every_reset():
    environment = ConveyorEnv(show_viewer=False, seed=1, belt_speed_m_per_s=0.123)
    try:
        for index in range(4):
            environment.reset(episode_index=index)
            assert environment.conveyor_speed_m_per_s == pytest.approx(0.123)
    finally:
        environment.close()


def test_randomization_can_be_switched_off():
    environment = ConveyorEnv(
        show_viewer=False,
        seed=2,
        randomize_belt_speed=False,
        randomize_layout=False,
    )
    try:
        for index in range(3):
            environment.reset(episode_index=index)
            assert environment.conveyor_speed_m_per_s == pytest.approx(0.18)
            assert np.all(environment.layout_offset_xy == 0.0)
            assert environment.layout.conveyor_center_xyz == (
                environment.nominal_layout.conveyor_center_xyz
            )
    finally:
        environment.close()


def test_narrow_speed_range_is_respected():
    environment = ConveyorEnv(
        show_viewer=False, seed=3, belt_speed_range=ValueRange(0.05, 0.06)
    )
    try:
        for index in range(5):
            environment.reset(episode_index=index)
            assert 0.05 <= environment.conveyor_speed_m_per_s <= 0.06
    finally:
        environment.close()


@pytest.mark.parametrize("speed", [0.02, 0.15, 0.30])
def test_belt_carries_the_cube_at_the_commanded_speed(speed):
    environment = ConveyorEnv(show_viewer=False, seed=4, belt_speed_m_per_s=speed)
    try:
        environment.reset(episode_index=0)
        home = environment.home_tool_pose
        # Let the cube reach belt speed, then measure over a clean interval.
        for _ in range(300):
            environment.step_task_space(home, environment.OPEN_GRIPPER_WIDTH_M)
        start_x, start_y = environment.object_pose[:2]
        start_t = environment.current_time
        for _ in range(1000):
            environment.step_task_space(home, environment.OPEN_GRIPPER_WIDTH_M)
        measured = (environment.object_pose[1] - start_y) / (
            environment.current_time - start_t
        )
        assert measured == pytest.approx(speed, rel=0.02)
        # It rides along without climbing or drifting sideways.
        assert environment.object_pose[2] == pytest.approx(
            environment.layout.conveyor_top_z + 0.025, abs=3e-3
        )
        assert abs(environment.object_pose[0] - start_x) < 2e-3
    finally:
        environment.close()


def test_a_stopped_belt_leaves_the_cube_where_it_spawned():
    environment = ConveyorEnv(show_viewer=False, seed=5, belt_speed_m_per_s=0.0)
    try:
        environment.reset(episode_index=0)
        start = environment.object_pose[:2].copy()
        home = environment.home_tool_pose
        for _ in range(2000):
            environment.step_task_space(home, environment.OPEN_GRIPPER_WIDTH_M)
        assert np.linalg.norm(environment.object_pose[:2] - start) < 1e-3
    finally:
        environment.close()


def test_gripper_command_moves_the_fingers(env):
    env.reset(episode_index=0)
    home = env.home_tool_pose
    for _ in range(1500):
        env.step_task_space(home, env.CLOSE_GRIPPER_WIDTH_M)
    assert env.gripper_width == pytest.approx(env.CLOSE_GRIPPER_WIDTH_M, abs=3e-3)
    assert env.gripper_width_cmd == pytest.approx(env.CLOSE_GRIPPER_WIDTH_M)
    for _ in range(1500):
        env.step_task_space(home, env.OPEN_GRIPPER_WIDTH_M)
    assert env.gripper_width == pytest.approx(env.OPEN_GRIPPER_WIDTH_M, abs=3e-3)


def test_step_task_space_rejects_a_malformed_pose(env):
    with pytest.raises(ValueError):
        env.step_task_space(np.zeros(6))


def test_tool_reaches_the_belt_and_the_bin():
    environment = ConveyorEnv(
        show_viewer=False, seed=6, belt_speed_m_per_s=0.0, randomize_layout=False
    )
    try:
        layout = environment.layout
        for goal in (
            (layout.conveyor_center_xyz[0], layout.conveyor_center_xyz[1], layout.conveyor_top_z + 0.02),
            (layout.conveyor_center_xyz[0], layout.conveyor_start_y + 0.15, layout.conveyor_top_z + 0.14),
            (layout.conveyor_center_xyz[0], layout.conveyor_end_y - 0.10, layout.conveyor_top_z + 0.14),
            (layout.target_bin_center_xyz[0], layout.target_bin_center_xyz[1], layout.target_bin_center_xyz[2] + 0.20),
            (layout.target_bin_center_xyz[0], layout.target_bin_center_xyz[1], layout.target_bin_center_xyz[2] + 0.08),
        ):
            environment.reset(episode_index=0)
            pose = _slew_to(environment, goal, seconds=6.0)
            assert np.linalg.norm(pose[:3] - np.array(goal)) < 5e-3, goal
            assert abs(abs(float(np.dot(pose[3:], GRASP_QUAT_WXYZ))) - 1.0) < 1e-3
    finally:
        environment.close()


def test_robot_does_not_collide_with_itself(env):
    """The contact channels leave robot-world and robot-cube pairs only."""
    env.reset(episode_index=0)
    home = env.home_tool_pose
    for _ in range(500):
        env.step_task_space(home, env.OPEN_GRIPPER_WIDTH_M)
    robot_root = int(env.model.body_rootid[env.model.body("ur5e/base").id])
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        roots = tuple(
            int(env.model.body_rootid[int(env.model.geom_bodyid[geom])])
            for geom in (contact.geom1, contact.geom2)
        )
        assert not (roots[0] == robot_root and roots[1] == robot_root)


def test_cube_properties_reach_the_compiled_model():
    properties = CubeProperties(mass_kg=0.25, half_extent_m=0.02, sliding_friction=0.7)
    environment = ConveyorEnv(
        show_viewer=False, seed=7, cube_properties=properties, belt_speed_m_per_s=0.0
    )
    try:
        assert float(environment.model.body_mass[environment.object_body_id]) == pytest.approx(0.25)
        collision = environment.model.geom("conveyor_cube/cube_collision")
        assert collision.size[0] == pytest.approx(0.02)
        assert collision.friction[0] == pytest.approx(0.7)
        # The belt's grip on the cube follows the cube's own friction.
        assert environment.belt_drive_friction == pytest.approx(0.7)
        environment.reset(episode_index=0)
        assert environment.object_pose[2] == pytest.approx(
            environment.layout.conveyor_top_z + 0.02, abs=3e-3
        )
    finally:
        environment.close()


def test_respawn_returns_the_cube_to_the_belt_start():
    environment = ConveyorEnv(
        show_viewer=False, seed=8, belt_speed_m_per_s=0.30, respawn_object=True
    )
    try:
        environment.reset(episode_index=0)
        home = environment.home_tool_pose
        for _ in range(9000):
            environment.step_task_space(home, environment.OPEN_GRIPPER_WIDTH_M)
            if environment.respawn_count:
                break
        assert environment.respawn_count >= 1
        assert environment.object_pose[1] < environment.layout.conveyor_start_y + 0.10
    finally:
        environment.close()


def test_render_returns_an_image(env):
    env.reset(episode_index=0)
    image = env.render(camera="third_person_camera", width=64, height=48)
    assert image.shape == (48, 64, 3)
    assert image.dtype == np.uint8
    assert image.max() > 0
