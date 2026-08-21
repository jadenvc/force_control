from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
TELEOP_DIR = Path(__file__).resolve().parents[1]
if str(TELEOP_DIR) not in sys.path:
    sys.path.insert(0, str(TELEOP_DIR))

from floating_cube_lift_teleop import (  # noqa: E402
    CUBE_COLOURS,
    CubeProperties,
    FloatingCubeLiftTeleop,
    sample_cube_properties,
)


class CubeLiftTest(unittest.TestCase):
    def test_default_cube_is_one_eighth_original_volume_at_constant_density(self) -> None:
        cube = CubeProperties()
        self.assertAlmostEqual(cube.size_m**3 / 0.055**3, 1.0 / 8.0)
        self.assertAlmostEqual(cube.mass_kg / 0.25, 1.0 / 8.0)
        self.assertAlmostEqual(cube.corner_radius_m / 0.006, 0.5)

    def test_default_mapping_and_sampling_cover_larger_workspace(self) -> None:
        np.testing.assert_allclose(FloatingCubeLiftTeleop.default_scale, 5.0)
        self.assertEqual(
            FloatingCubeLiftTeleop.default_cam_name, "wsg50/d435i/rgb"
        )
        np.testing.assert_allclose(
            FloatingCubeLiftTeleop.default_start_prism, [0.10, 0.10, 0.05]
        )
        with FloatingCubeLiftTeleop() as env:
            self.assertIn(
                FloatingCubeLiftTeleop.default_cam_name, env.camera_names()
            )
            np.testing.assert_allclose(
                env.workspace_high - env.workspace_low, [0.40, 0.40, 0.395]
            )

    def test_randomization_and_rounded_collision_stay_consistent(self) -> None:
        base = CubeProperties()
        for seed in range(20):
            properties = sample_cube_properties(
                base, np.random.default_rng(seed), size_jitter=0.1, mass_jitter=0.2
            )
            self.assertLessEqual(abs(properties.size_m / base.size_m - 1.0), 0.1)
            self.assertLessEqual(abs(properties.mass_kg / base.mass_kg - 1.0), 0.2)
            self.assertAlmostEqual(
                properties.corner_radius_m / properties.size_m,
                base.corner_radius_m / base.size_m,
            )

        with FloatingCubeLiftTeleop(physical_properties=base) as env:
            radius = env.model.geom_margin[env.book_collision_geom_id]
            inner = env.model.geom_size[env.book_collision_geom_id]
            np.testing.assert_allclose(2.0 * (inner + radius), base.size_m)
            self.assertGreater(
                env.model.mesh_vertnum[env.book_mesh_id],
                300,
            )

    def test_scripted_grasp_is_force_limited_and_lifts_cube(self) -> None:
        with FloatingCubeLiftTeleop(
            physical_properties=CubeProperties(),
            grasp_force_limit=25.0,
            tip_softness=0.5,
            success_height=0.08,
        ) as env:
            actuator_force = []
            grasp_force = []

            def move_to(goal, speed, gripper, max_steps=2000, stop_on_success=False):
                target = env.drive_target.copy()
                for _ in range(max_steps):
                    delta = np.asarray(goal) - target
                    distance = np.linalg.norm(delta)
                    if distance > 1e-6:
                        target += delta * min(
                            1.0,
                            speed * env.timestep / distance,
                        )
                    env.set_gripper_command(gripper)
                    env.step(target)
                    actuator_force.append(abs(env.gripper_actuator_force))
                    grasp_force.append(env.grasp_force())
                    if stop_on_success and env.success():
                        return target
                    if distance <= 2e-4:
                        return target
                self.fail("scripted move did not converge")

            move_to(env.scene["engage"], 0.14, env.gripper_open_command)
            for _ in range(450):
                env.set_gripper_command(0.0)
                env.step(env.scene["engage"])
                actuator_force.append(abs(env.gripper_actuator_force))
                grasp_force.append(env.grasp_force())
            move_to(env.scene["waypoints"][-1], 0.14, 0.0, stop_on_success=True)

            self.assertTrue(env.success())
            self.assertGreater(max(grasp_force), 5.0)
            self.assertLessEqual(max(actuator_force), 25.0 + 1e-6)
            self.assertLess(env.gripper_opening, env.gripper_open_command)

    def test_table_limit_settles_smoothly_at_requested_force(self) -> None:
        with FloatingCubeLiftTeleop(
            surface_force_limit=40.0,
            tip_softness=0.5,
            force_sensor_cutoff_hz=30.0,
        ) as env:
            target = env.tool_home.copy()
            goal = np.array([0.43, 0.10, env.workspace_low[2]])
            force = []
            for _ in range(2500):
                delta = goal - target
                distance = np.linalg.norm(delta)
                if distance > 0.0:
                    target += delta * min(1.0, 0.20 * env.timestep / distance)
                env.step(target)
                force.append(np.linalg.norm(env.sensor_wrench("world")[:3]))
            steady = np.asarray(force[-500:])
            self.assertTrue(env.surface_limit_active)
            self.assertAlmostEqual(float(np.mean(steady)), 40.0, delta=2.0)
            self.assertLess(float(np.std(steady)), 0.25)

    def test_workspace_clamps_requested_tool_target(self) -> None:
        with FloatingCubeLiftTeleop() as env:
            outside = env.workspace_high + 1.0
            bounded = env.limited_target(outside)
            self.assertTrue(env.workspace_limit_active)
            self.assertTrue(np.all(bounded <= env.workspace_high + 1e-12))


if __name__ == "__main__":
    unittest.main()
