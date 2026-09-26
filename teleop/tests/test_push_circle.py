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

from push_circle_teleop import (  # noqa: E402
    PushCircleProperties,
    PushCircleTeleop,
    circle_overlap_area,
)


class CircleOverlapAreaTest(unittest.TestCase):
    def test_no_overlap_beyond_sum_of_radii(self) -> None:
        self.assertEqual(circle_overlap_area(0.2, 0.05, 0.06), 0.0)

    def test_full_containment_matches_smaller_circles_area(self) -> None:
        area = circle_overlap_area(0.0, 0.05, 0.06)
        self.assertAlmostEqual(area, np.pi * 0.05**2, places=9)

    def test_exact_tangency_is_zero(self) -> None:
        self.assertAlmostEqual(circle_overlap_area(0.11, 0.05, 0.06), 0.0, places=9)


class PushCircleModelPropertiesTest(unittest.TestCase):
    def test_rejects_nonpositive_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            PushCircleProperties(object_mass_kg=0.0)
        with self.assertRaises(ValueError):
            PushCircleProperties(pusher_radius_m=-0.01)

    def test_rejects_negative_friction(self) -> None:
        with self.assertRaises(ValueError):
            PushCircleProperties(table_friction=-0.1)


class PushCircleTeleopTest(unittest.TestCase):
    def test_free_space_contact_force_is_exactly_zero(self) -> None:
        with PushCircleTeleop() as env:
            env.reset(object_xy=(0.0, 0.0), pusher_xy=(-0.3, -0.3))
            self.assertTrue(np.all(env.pusher_contact_force_xy() == 0.0))

    def test_unreachable_success_threshold_raises(self) -> None:
        props = PushCircleProperties(object_radius_m=0.03, goal_radius_m=0.06)
        with self.assertRaises(ValueError):
            PushCircleTeleop(properties=props, success_threshold=0.5)

    def test_coverage_is_one_at_the_goal_and_zero_far_away(self) -> None:
        with PushCircleTeleop(goal_xy=(0.10, 0.0)) as env:
            env.reset(object_xy=(0.10, 0.0), pusher_xy=(-0.10, 0.0))
            self.assertGreater(env.coverage_fraction(), 0.999)
            self.assertTrue(env.success())

            env.reset(object_xy=(1.0, 1.0), pusher_xy=(-0.10, 0.0))
            self.assertLess(env.coverage_fraction(), 1e-6)
            self.assertFalse(env.success())

    def test_higher_table_friction_reduces_post_push_coast_distance(self) -> None:
        def coast_distance(table_friction):
            props = PushCircleProperties(table_friction=table_friction, pusher_friction=0.8)
            env = PushCircleTeleop(properties=props, pusher_kp=800.0)
            env.reset(object_xy=(0.0, 0.0), pusher_xy=(-0.09, 0.0))
            target = np.array([-0.09, 0.0])
            for _ in range(150):
                target[0] += 0.001
                env.step(target, n_substeps=1)
            pose_at_release = env.object_pos.copy()
            held_target = env.pusher_pos.copy()
            for _ in range(1000):
                env.step(held_target, n_substeps=1)
            pose_after_coast = env.object_pos.copy()
            return float(np.linalg.norm(pose_after_coast - pose_at_release))

        low_friction_slide = coast_distance(0.03)
        high_friction_slide = coast_distance(1.5)
        self.assertGreater(low_friction_slide, 10.0 * high_friction_slide)

    def test_disturbance_moves_object_with_no_contact(self) -> None:
        with PushCircleTeleop(
            seed=0, disturbance_force_n=1.0, disturbance_tau_s=0.5
        ) as env:
            env.reset(object_xy=(0.0, 0.0), pusher_xy=(-0.3, -0.3))
            for _ in range(3000):
                env.step(env.pusher_pos, n_substeps=1)
            self.assertGreater(np.linalg.norm(env.object_pos), 0.01)

    def test_goal_move_relocates_and_stays_within_bounds(self) -> None:
        with PushCircleTeleop(
            seed=0,
            goal_move_min_interval_s=0.05,
            goal_move_max_interval_s=0.1,
            goal_move_xy_half_m=0.15,
        ) as env:
            env.reset(object_xy=(0.0, 0.0), pusher_xy=(-0.3, -0.3))
            initial = env.goal_xy.copy()
            moved = False
            for _ in range(2000):
                env.step(env.pusher_pos, n_substeps=1)
                if not np.array_equal(env.goal_xy, initial):
                    moved = True
                self.assertTrue(np.all(np.abs(env.goal_xy) <= 0.15 + 1e-9))
            self.assertTrue(moved)

    def test_reset_and_randomize_start_stay_finite_and_non_overlapping(self) -> None:
        with PushCircleTeleop(seed=1) as env:
            for _ in range(200):
                env.randomize_start(center_probability=0.9)
                self.assertTrue(np.all(np.isfinite(env.object_pos)))
                self.assertTrue(np.all(np.isfinite(env.pusher_pos)))
                self.assertFalse(env._pusher_overlaps_object())


if __name__ == "__main__":
    unittest.main()
