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
            self.assertEqual(
                int(model.geom_type[geom_id]),
                int(mujoco.mjtGeom.mjGEOM_MESH),
            )
            np.testing.assert_allclose(
                model.geom_pos[geom_id],
                [0.09, 0.0425, 0.01],
            )
            np.testing.assert_allclose(
                model.geom_friction[geom_id],
                properties.friction,
            )
            collision_mesh_id = int(model.geom_dataid[geom_id])
            self.assertGreaterEqual(int(model.mesh_graphadr[collision_mesh_id]), 0)
            collision_vertex_start = int(model.mesh_vertadr[collision_mesh_id])
            collision_vertex_count = int(model.mesh_vertnum[collision_mesh_id])
            collision_vertices = model.mesh_vert[
                collision_vertex_start : collision_vertex_start + collision_vertex_count
            ]
            np.testing.assert_allclose(
                sorted(np.ptp(collision_vertices, axis=0)),
                sorted(
                    [
                        properties.length_m,
                        properties.width_m,
                        properties.thickness_m,
                    ]
                ),
                atol=1e-7,
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

    def test_contact_allowlist_separates_book_and_gripper_surface_contacts(
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
            guard_geom_id = model.geom(
                "bookend2_blender/robot_floor_guard"
            ).id
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
            support_geom_ids = {
                model.geom(f"bookend2_blender/{name}").id
                for name in ("book_wall", "book_pivot", "book_floor")
            }

            def can_collide(first: int, second: int) -> bool:
                return bool(
                    int(model.geom_contype[first])
                    & int(model.geom_conaffinity[second])
                    or int(model.geom_contype[second])
                    & int(model.geom_conaffinity[first])
                )

            # The object and the robot-only floor guard emit independent bits.
            # No explicit pair can bypass the masks.
            self.assertEqual(int(model.geom_contype[object_geom_id]), 1)
            self.assertEqual(int(model.geom_conaffinity[object_geom_id]), 0)
            self.assertEqual(int(model.geom_contype[guard_geom_id]), 2)
            self.assertEqual(int(model.geom_conaffinity[guard_geom_id]), 0)
            self.assertEqual(
                int(model.geom_type[guard_geom_id]),
                int(mujoco.mjtGeom.mjGEOM_BOX),
            )
            self.assertEqual(int(model.geom_condim[guard_geom_id]), 3)
            self.assertEqual(int(model.geom_priority[guard_geom_id]), 20)
            self.assertAlmostEqual(float(model.geom_margin[guard_geom_id]), 0.0005)
            np.testing.assert_allclose(
                model.geom_friction[guard_geom_id],
                [0.01, 0.0, 0.0],
            )
            np.testing.assert_allclose(
                model.geom_solref[guard_geom_id],
                [0.015, 2.0],
            )
            self.assertEqual(model.nexclude, 0)
            self.assertEqual(model.npair, 0)

            saw_robot = False
            saw_support = False
            saw_gripper_guard = False
            for geom_id in range(model.ngeom):
                contype = int(model.geom_contype[geom_id])
                conaffinity = int(model.geom_conaffinity[geom_id])
                if geom_id in (object_geom_id, guard_geom_id):
                    continue
                self.assertEqual(contype, 0)
                if conaffinity == 0:
                    continue
                body_id = int(model.geom_bodyid[geom_id])
                if body_id in robot_body_ids:
                    saw_robot = True
                    self.assertTrue(can_collide(object_geom_id, geom_id))
                    if body_id in gripper_body_ids:
                        self.assertEqual(conaffinity, 3)
                        self.assertTrue(can_collide(guard_geom_id, geom_id))
                        saw_gripper_guard = True
                    else:
                        self.assertEqual(conaffinity, 1)
                        self.assertFalse(can_collide(guard_geom_id, geom_id))
                elif body_id == support_body_id:
                    self.assertIn(geom_id, support_geom_ids)
                    self.assertEqual(conaffinity, 1)
                    self.assertTrue(can_collide(object_geom_id, geom_id))
                    self.assertFalse(can_collide(guard_geom_id, geom_id))
                    saw_support = True
                else:
                    self.fail(f"unexpected allowed body {model.body(body_id).name}")

            self.assertTrue(saw_robot)
            self.assertTrue(saw_support)
            self.assertTrue(saw_gripper_guard)
            self.assertFalse(can_collide(object_geom_id, guard_geom_id))

    def test_fingertips_use_one_rounded_contact_geom_per_finger(self) -> None:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = (0.4, 0.0, 0.2)

        with FlipUpEnv(transform, transform, show_viewer=False) as environment:
            model = environment.model
            for side in ("right", "left"):
                geom_id = model.geom(f"ur5e/wsg50/{side}_tip_pad").id
                self.assertEqual(
                    int(model.geom_type[geom_id]),
                    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                )
                np.testing.assert_allclose(model.geom_size[geom_id, :2], [0.008, 0.008])
                self.assertEqual(int(model.geom_condim[geom_id]), 4)

                finger_body_id = model.body(f"ur5e/wsg50/{side}_finger").id
                physical_geoms = [
                    geom_id
                    for geom_id in range(model.ngeom)
                    if int(model.geom_bodyid[geom_id]) == finger_body_id
                    and int(model.geom_conaffinity[geom_id]) != 0
                ]
                mesh_geoms = [
                    geom_id
                    for geom_id in physical_geoms
                    if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
                ]
                self.assertEqual(mesh_geoms, [])


if __name__ == "__main__":
    unittest.main()
