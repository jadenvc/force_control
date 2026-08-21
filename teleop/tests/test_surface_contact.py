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

from floating_flipup_teleop import FloatingFlipUpTeleop  # noqa: E402


class FloatingSurfaceContactTest(unittest.TestCase):
    def test_visible_table_and_bookend_surfaces_have_distinct_contact_bits(self):
        with FloatingFlipUpTeleop(seed=0) as environment:
            model = environment.model
            book_id = environment.book_collision_geom_id
            table_id = model.geom("table/table_surface").id
            support_ids = {
                model.geom(f"bookend2_blender/book_{part}").id
                for part in ("wall", "pivot", "floor")
            }
            robot_surface_ids = {
                model.geom(f"bookend2_blender/robot_{part}_surface").id
                for part in ("wall", "pivot", "floor")
            }

            def can_collide(first, second):
                return bool(
                    int(model.geom_contype[first])
                    & int(model.geom_conaffinity[second])
                    or int(model.geom_contype[second])
                    & int(model.geom_conaffinity[first])
                )

            gripper_geoms = [
                geom_id
                for geom_id in range(model.ngeom)
                if model.geom(geom_id).name.startswith("wsg50/")
                and not model.geom(geom_id).name.startswith("wsg50/d435i/")
                and int(model.geom_conaffinity[geom_id]) != 0
            ]
            self.assertTrue(gripper_geoms)
            self.assertEqual(int(model.geom_contype[book_id]), 1)
            self.assertEqual(int(model.geom_contype[table_id]), 4)
            for geom_id in gripper_geoms:
                self.assertTrue(can_collide(book_id, geom_id))
                self.assertTrue(can_collide(table_id, geom_id))
                for surface_id in robot_surface_ids:
                    self.assertTrue(can_collide(surface_id, geom_id))
            for support_id in support_ids:
                self.assertTrue(can_collide(book_id, support_id))
                for surface_id in robot_surface_ids:
                    self.assertFalse(can_collide(surface_id, support_id))
            self.assertFalse(can_collide(book_id, table_id))

    def test_table_contact_is_bounded_and_releases(self):
        with FloatingFlipUpTeleop(seed=0) as environment:
            target = environment.tool_home.copy()
            goals = (
                np.array([0.15, -0.25, 0.18]),
                np.array([0.15, -0.25, 0.00]),
            )
            goal_index = 0
            saw_table_contact = False
            maximum_force = 0.0

            for _ in range(3500):
                goal = goals[goal_index]
                delta = goal - target
                distance = np.linalg.norm(delta)
                if distance > 1e-12:
                    target += delta * min(
                        1.0, 0.25 * environment.timestep / distance
                    )
                environment.step(target)
                maximum_force = max(
                    maximum_force,
                    float(np.linalg.norm(environment.contact_force())),
                )
                saw_table_contact |= any(
                    "table/table_surface"
                    in (
                        environment.model.geom(
                            environment.data.contact[index].geom1
                        ).name,
                        environment.model.geom(
                            environment.data.contact[index].geom2
                        ).name,
                    )
                    for index in range(environment.data.ncon)
                )
                if distance < 2e-4 and goal_index == 0:
                    goal_index = 1

            self.assertEqual(goal_index, 1)
            self.assertTrue(saw_table_contact)
            self.assertTrue(environment.surface_limit_active)
            self.assertGreater(environment.tool_pos[2], 0.05)
            self.assertGreater(environment.contact_force()[2], 60.0)
            self.assertLess(maximum_force, 90.0)
            self.assertGreater(
                np.linalg.norm(environment.requested_target - environment.drive_target),
                0.03,
            )

            release = goals[0]
            for _ in range(1200):
                delta = release - target
                distance = np.linalg.norm(delta)
                if distance > 1e-12:
                    target += delta * min(
                        1.0, 0.25 * environment.timestep / distance
                    )
                environment.step(target)

            self.assertFalse(environment.surface_limit_active)
            self.assertLess(np.linalg.norm(environment.contact_force()), 0.1)
            self.assertLess(np.linalg.norm(environment.tool_pos - release), 1e-3)


if __name__ == "__main__":
    unittest.main()
