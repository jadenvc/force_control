from __future__ import annotations

import unittest

import mujoco
import numpy as np

from flipup.environment import FlipUpEnv
from flipup.physical_properties import PhysicalProperties


class EnvironmentPhysicalPropertiesTest(unittest.TestCase):
    def test_compiled_book_matches_requested_properties(self) -> None:
        properties = PhysicalProperties(
            mass_kg=0.73,
            sliding_friction=0.27,
            torsional_friction=0.004,
            rolling_friction=0.0006,
            length_m=0.18,
            width_m=0.085,
            thickness_m=0.02,
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = (0.4, 0.0, 0.2)

        with FlipUpEnv(
            transform,
            transform,
            show_viewer=False,
            physical_properties=properties,
        ) as environment:
            model = environment.model
            geom_id = environment.book_collision_geom_id

            self.assertAlmostEqual(
                model.body_mass[environment.book_body_id],
                properties.mass_kg,
            )
            np.testing.assert_allclose(
                model.geom_size[geom_id],
                [0.09, 0.0425, 0.01],
            )
            np.testing.assert_allclose(
                model.geom_pos[geom_id],
                [0.09, 0.0425, 0.01],
            )
            np.testing.assert_allclose(
                model.geom_friction[geom_id],
                properties.friction,
            )

            visual_geom_id = model.geom("book2_blend/book_visual").id
            mesh_id = int(model.geom_dataid[visual_geom_id])
            vertex_start = int(model.mesh_vertadr[mesh_id])
            vertex_count = int(model.mesh_vertnum[mesh_id])
            vertices = model.mesh_vert[vertex_start : vertex_start + vertex_count]
            np.testing.assert_allclose(
                sorted(np.ptp(vertices, axis=0)),
                sorted(
                    [
                        properties.length_m,
                        properties.width_m,
                        properties.thickness_m,
                    ]
                ),
                atol=1e-7,
            )

    def test_contact_allowlist_separates_book_support_and_visible_surfaces(
        self,
    ) -> None:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = (0.4, 0.0, 0.2)

        with FlipUpEnv(transform, transform, show_viewer=False) as environment:
            model = environment.model
            object_geom_id = environment.book_collision_geom_id
            support_body_id = model.body(
                "bookend2_blender/bookend2_blender"
            ).id
            table_geom_id = model.geom("table/table_surface").id
            robot_surface_ids = {
                model.geom(f"bookend2_blender/robot_{part}_surface").id
                for part in ("wall", "pivot", "floor")
            }
            support_ids = {
                model.geom(f"bookend2_blender/book_{part}").id
                for part in ("wall", "pivot", "floor")
            }
            robot_body_ids = {
                body_id
                for body_id in range(model.nbody)
                if model.body(body_id).name.startswith("ur5e/")
            }

            gripper_body_ids = {
                body_id
                for body_id in robot_body_ids
                if model.body(body_id).name.startswith("ur5e/wsg50/")
                and not model.body(body_id).name.startswith("ur5e/wsg50/d435i/")
            }

            def can_collide(first: int, second: int) -> bool:
                return bool(
                    int(model.geom_contype[first])
                    & int(model.geom_conaffinity[second])
                    or int(model.geom_contype[second])
                    & int(model.geom_conaffinity[first])
                )

            self.assertEqual(int(model.geom_contype[object_geom_id]), 1)
            self.assertEqual(int(model.geom_conaffinity[object_geom_id]), 0)
            self.assertEqual(int(model.geom_contype[table_geom_id]), 4)
            self.assertEqual(int(model.geom_conaffinity[table_geom_id]), 0)
            np.testing.assert_allclose(model.geom_size[table_geom_id], [0.7, 1.0, 0.05])
            self.assertEqual(model.nexclude, 0)
            self.assertEqual(model.npair, 0)

            saw_robot = False
            saw_support = False
            for geom_id in range(model.ngeom):
                contype = int(model.geom_contype[geom_id])
                conaffinity = int(model.geom_conaffinity[geom_id])
                if geom_id in {object_geom_id, table_geom_id, *robot_surface_ids}:
                    continue
                self.assertEqual(contype, 0)
                if conaffinity == 0:
                    continue
                body_id = int(model.geom_bodyid[geom_id])
                if body_id in robot_body_ids:
                    saw_robot = True
                    self.assertIn(conaffinity, (5, 7))
                    self.assertTrue(can_collide(object_geom_id, geom_id))
                    self.assertTrue(can_collide(table_geom_id, geom_id))
                    if body_id in gripper_body_ids:
                        self.assertEqual(conaffinity, 7)
                        for surface_id in robot_surface_ids:
                            self.assertTrue(can_collide(surface_id, geom_id))
                elif body_id == support_body_id:
                    self.assertIn(geom_id, support_ids)
                    self.assertEqual(conaffinity, 1)
                    self.assertTrue(can_collide(object_geom_id, geom_id))
                    saw_support = True
                else:
                    self.fail(f"unexpected allowed body {model.body(body_id).name}")

            self.assertTrue(saw_robot)
            self.assertTrue(saw_support)
            for surface_id in robot_surface_ids:
                self.assertEqual(int(model.geom_contype[surface_id]), 2)
                self.assertEqual(int(model.geom_conaffinity[surface_id]), 0)
                self.assertFalse(can_collide(object_geom_id, surface_id))
                self.assertFalse(can_collide(table_geom_id, surface_id))

    def test_fingers_use_smooth_visual_aligned_collision_envelopes(self) -> None:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = (0.4, 0.0, 0.2)
        with FlipUpEnv(transform, transform, show_viewer=False) as environment:
            model = environment.model
            for side in ("right", "left"):
                guard_id = model.geom(f"ur5e/wsg50/{side}_finger_guard").id
                tip_id = model.geom(f"ur5e/wsg50/{side}_tip_pad").id
                self.assertEqual(
                    int(model.geom_type[guard_id]),
                    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
                )
                self.assertEqual(
                    int(model.geom_type[tip_id]),
                    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                )
                np.testing.assert_allclose(model.geom_size[guard_id], [0.014, 0.021, 0.055])
                np.testing.assert_allclose(model.geom_size[tip_id, :2], [0.008, 0.008])


if __name__ == "__main__":
    unittest.main()
