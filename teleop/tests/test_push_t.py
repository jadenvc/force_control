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

from push_t_teleop import PushTProperties, PushTTeleop  # noqa: E402


class PushTPropertiesTest(unittest.TestCase):
    def test_rejects_nonpositive_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            PushTProperties(t_mass_kg=0.0)
        with self.assertRaises(ValueError):
            PushTProperties(pusher_radius_m=-0.01)

    def test_rejects_negative_friction(self) -> None:
        with self.assertRaises(ValueError):
            PushTProperties(table_friction=-0.1)
        with self.assertRaises(ValueError):
            PushTProperties(pusher_friction=-0.1)


class PushTTeleopTest(unittest.TestCase):
    def test_free_space_contact_force_is_exactly_zero(self) -> None:
        with PushTTeleop() as env:
            env.reset(t_pose=(0.0, 0.0, 0.0), pusher_pos=(-0.3, -0.3))
            self.assertEqual(env.pusher_contact_force(), 0.0)

    def test_table_contact_generates_the_full_weight_as_normal_force(self) -> None:
        props = PushTProperties(table_friction=0.4)
        with PushTTeleop(properties=props) as env:
            env.reset(t_pose=(0.0, 0.0, 0.0), pusher_pos=(-0.3, -0.3))
            total_normal = 0.0
            buf = np.zeros(6)
            import mujoco

            for index in range(env.data.ncon):
                mujoco.mj_contactForce(env.model.ptr, env.data.ptr, index, buf)
                total_normal += buf[0]
            self.assertAlmostEqual(
                total_normal, props.t_mass_kg * 9.81, delta=0.01
            )

    def test_coverage_is_one_at_the_goal_pose_and_zero_far_away(self) -> None:
        with PushTTeleop(goal_pose=(0.10, 0.0, 0.0)) as env:
            env.reset(t_pose=(0.10, 0.0, 0.0), pusher_pos=(-0.10, 0.0))
            self.assertGreater(env.coverage_fraction(), 0.999)
            self.assertTrue(env.success())

            env.reset(t_pose=(1.0, 1.0, 0.0), pusher_pos=(-0.10, 0.0))
            self.assertLess(env.coverage_fraction(), 1e-6)
            self.assertFalse(env.success())

    def test_higher_table_friction_reduces_post_push_coast_distance(self) -> None:
        def coast_distance(table_friction):
            props = PushTProperties(table_friction=table_friction, pusher_friction=0.8)
            env = PushTTeleop(properties=props, pusher_kp=800.0)
            env.reset(t_pose=(0.0, 0.0, 0.0), pusher_pos=(-0.09, 0.0))
            target = np.array([-0.09, 0.0])
            for _ in range(150):
                target[0] += 0.001
                env.step(target, n_substeps=1)
            pose_at_release = env.t_pose[:2].copy()
            held_target = env.pusher_pos.copy()
            for _ in range(1000):
                env.step(held_target, n_substeps=1)
            pose_after_coast = env.t_pose[:2].copy()
            return float(np.linalg.norm(pose_after_coast - pose_at_release))

        low_friction_slide = coast_distance(0.03)
        high_friction_slide = coast_distance(1.5)
        self.assertGreater(low_friction_slide, 10.0 * high_friction_slide)

    def test_off_center_push_rotates_the_t(self) -> None:
        props = PushTProperties(table_friction=0.4, pusher_friction=0.8)
        with PushTTeleop(properties=props, pusher_kp=800.0) as env:
            env.reset(t_pose=(0.0, 0.0, 0.0), pusher_pos=(-0.03, -0.09))
            target = np.array([-0.03, -0.09])
            for _ in range(400):
                target[1] += 0.0006
                env.step(target, n_substeps=1)
            self.assertGreater(abs(env.t_pose[2]), 0.05)

    def test_reset_and_randomize_start_stay_finite(self) -> None:
        with PushTTeleop(seed=1) as env:
            env.randomize_start()
            self.assertTrue(np.all(np.isfinite(env.t_pose)))
            self.assertTrue(np.all(np.isfinite(env.pusher_pos)))


if __name__ == "__main__":
    unittest.main()
