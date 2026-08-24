from __future__ import annotations

import numpy as np
import pytest

from conveyor.environment import ConveyorEnv
from conveyor.heuristic import ConveyorHeuristic, run_conveyor
from conveyor.judge import ConveyorJudge

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def test_judge_reports_a_miss_when_the_cube_runs_off_the_belt():
    env = ConveyorEnv(show_viewer=False, seed=0, belt_speed_m_per_s=0.30)
    try:
        env.reset(episode_index=0)
        judge = ConveyorJudge()
        judge.reset()
        home = env.home_tool_pose
        while not judge.done(env):
            env.step_task_space(home, env.OPEN_GRIPPER_WIDTH_M)
            judge.update(env)
        assert judge.object_missed
        assert not judge.success
        assert judge.termination_reason == "object_missed"
        # 1.2 m of belt at 0.3 m/s, so the miss lands well inside the time limit.
        assert env.current_time < judge.time_limit_s
    finally:
        env.close()


def test_judge_reports_the_time_limit_when_the_belt_is_stopped():
    env = ConveyorEnv(show_viewer=False, seed=0, belt_speed_m_per_s=0.0)
    try:
        env.reset(episode_index=0)
        judge = ConveyorJudge(time_limit_s=1.0)
        judge.reset()
        home = env.home_tool_pose
        while not judge.done(env):
            env.step_task_space(home, env.OPEN_GRIPPER_WIDTH_M)
            judge.update(env)
        assert judge.time_limit_reached
        assert judge.termination_reason == "time_limit"
        assert not judge.success
    finally:
        env.close()


def test_heuristic_estimates_the_belt_speed_from_observations_alone():
    env = ConveyorEnv(show_viewer=False, seed=0, belt_speed_m_per_s=0.22)
    try:
        env.reset(episode_index=0)
        agent = ConveyorHeuristic(env)
        for _ in range(1200):
            target_pose, width = agent.step()
            env.step_task_space(target_pose, width)
        assert agent.estimated_speed_m_per_s == pytest.approx(0.22, rel=0.15)
    finally:
        env.close()


def test_heuristic_commands_are_slew_limited():
    env = ConveyorEnv(show_viewer=False, seed=0)
    try:
        env.reset(episode_index=0)
        agent = ConveyorHeuristic(env)
        previous = agent.target_pose[:3].copy()
        limit = agent.position_speed_m_per_s * env.timestep + 1e-9
        for _ in range(3000):
            target_pose, width = agent.step()
            assert np.linalg.norm(target_pose[:3] - previous) <= limit
            previous = target_pose[:3].copy()
            env.step_task_space(target_pose, width)
    finally:
        env.close()


@pytest.mark.parametrize("episode_index", [0, 1, 2])
def test_scripted_episodes_succeed_with_randomized_speed(episode_index):
    result = run_conveyor(
        seed=0,
        episode_index=episode_index,
        show_viewer=False,
        verbose=False,
    )
    assert result.success, result.termination_reason
    assert result.picked_up
    assert 0.01 <= result.belt_speed_m_per_s <= 0.30
    assert result.estimated_belt_speed_m_per_s == pytest.approx(
        result.belt_speed_m_per_s, abs=0.05
    )


@pytest.mark.parametrize("speed", [0.01, 0.15, 0.30])
def test_scripted_episodes_succeed_across_the_speed_range(speed):
    result = run_conveyor(
        seed=100,
        episode_index=0,
        show_viewer=False,
        verbose=False,
        belt_speed_m_per_s=speed,
    )
    assert result.success, f"{speed} m/s: {result.termination_reason}"
    assert result.belt_speed_m_per_s == pytest.approx(speed)


def test_run_conveyor_rejects_conflicting_cube_arguments():
    from conveyor.properties import DEFAULT_CUBE_PROPERTIES

    with pytest.raises(ValueError):
        run_conveyor(
            show_viewer=False,
            verbose=False,
            cube_properties=DEFAULT_CUBE_PROPERTIES,
            randomize_cube=True,
        )
