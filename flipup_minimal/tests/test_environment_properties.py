from __future__ import annotations

import unittest

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

    def test_contact_allowlist_contains_only_robot_object_and_object_support(
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
            robot_body_ids = {
                body_id
                for body_id in range(model.nbody)
                if model.body(body_id).name.startswith("ur5e/")
            }

            # The object emits one collision bit and only robot/support
            # collision geoms accept it. No explicit pairs can bypass this
            # mask, and no other geom can initiate or accept contact.
            self.assertEqual(int(model.geom_contype[object_geom_id]), 1)
            self.assertEqual(int(model.geom_conaffinity[object_geom_id]), 0)
            self.assertEqual(model.nexclude, 0)
            self.assertEqual(model.npair, 0)

            saw_robot = False
            saw_support = False
            for geom_id in range(model.ngeom):
                contype = int(model.geom_contype[geom_id])
                conaffinity = int(model.geom_conaffinity[geom_id])
                if geom_id == object_geom_id:
                    continue
                self.assertEqual(contype, 0)
                if conaffinity == 0:
                    continue
                self.assertEqual(conaffinity, 1)
                body_id = int(model.geom_bodyid[geom_id])
                if body_id in robot_body_ids:
                    saw_robot = True
                elif body_id == support_body_id:
                    saw_support = True
                else:
                    self.fail(f"unexpected allowed body {model.body(body_id).name}")

            self.assertTrue(saw_robot)
            self.assertTrue(saw_support)


if __name__ == "__main__":
    unittest.main()
