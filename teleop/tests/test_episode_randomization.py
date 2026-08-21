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

from flipup_teleop import (
    FlipUpTeleop,
    flipup_scene,
    sample_book_color,
    sample_episode_properties,
    sample_start_pose,
)
from floating_flipup_teleop import FloatingFlipUpTeleop, advance_two_pole_filter
from flipup.physical_properties import DEFAULT_PHYSICAL_PROPERTIES


class EpisodeRandomizationTest(unittest.TestCase):
    def test_book_dimensions_mass_and_start_stay_in_requested_bounds(self) -> None:
        base = DEFAULT_PHYSICAL_PROPERTIES
        for seed in range(40):
            rng = np.random.default_rng(seed)
            properties = sample_episode_properties(base, rng)
            self.assertLessEqual(
                abs(properties.length_m / base.length_m - 1.0), 0.20 + 1e-12
            )
            self.assertLessEqual(
                abs(properties.width_m / base.width_m - 1.0), 0.20 + 1e-12
            )
            self.assertLessEqual(
                abs(properties.mass_kg / base.mass_kg - 1.0), 0.20 + 1e-12
            )

            scene = flipup_scene(0, properties)
            _, sample = sample_start_pose(
                scene, rng, prism_size=(0.04, 0.06, 0.05)
            )
            normalized = np.asarray(sample["normalized_depth_lateral_vertical"])
            self.assertTrue(np.all(np.abs(normalized) <= 0.5 + 1e-12))

    def test_first_start_is_exact_prism_center(self) -> None:
        scene = flipup_scene(0)
        position, sample = sample_start_pose(
            scene, np.random.default_rng(1), force_center=True
        )
        np.testing.assert_allclose(position, scene["prepare"])
        self.assertEqual(sample["component"], "center")

    def test_runtime_configuration_changes_mass_geometry_color_and_start(self) -> None:
        rng = np.random.default_rng(12)
        properties = sample_episode_properties(DEFAULT_PHYSICAL_PROPERTIES, rng)
        color = sample_book_color(rng)
        scene = flipup_scene(0, properties)
        start, _ = sample_start_pose(scene, rng)
        with FlipUpTeleop(seed=0, settle_s=0.0) as env:
            env.configure_episode(properties, color, start)
            self.assertAlmostEqual(
                env.model.body_mass[env.book_body_id], properties.mass_kg
            )
            np.testing.assert_allclose(
                env.model.geom_size[env.book_collision_geom_id],
                [
                    properties.length_m / 2.0,
                    properties.width_m / 2.0,
                    properties.thickness_m / 2.0,
                ],
            )
            np.testing.assert_allclose(
                env.model.geom_rgba[env.book_visual_geom_id], color
            )
            np.testing.assert_allclose(env.tool_home, start)

    def test_collision_bounds_remain_at_maximum_randomized_envelope(self) -> None:
        base = DEFAULT_PHYSICAL_PROPERTIES
        envelope = np.array(
            [1.2 * base.length_m, 1.2 * base.width_m, base.thickness_m]
        )
        with FloatingFlipUpTeleop(
            seed=0, collision_envelope_dimensions=envelope
        ) as env:
            compiled_aabb = env.model.geom_aabb[env.book_collision_geom_id].copy()
            compiled_bvh = env.model.bvh_aabb.copy()
            for scale in (0.8, 1.2):
                properties = type(base)(
                    mass_kg=base.mass_kg,
                    sliding_friction=base.sliding_friction,
                    torsional_friction=base.torsional_friction,
                    rolling_friction=base.rolling_friction,
                    length_m=scale * base.length_m,
                    width_m=scale * base.width_m,
                    thickness_m=base.thickness_m,
                )
                env.configure_episode(
                    properties,
                    np.array([0.4, 0.2, 0.1, 1.0]),
                    flipup_scene(0, properties)["prepare"],
                )
                np.testing.assert_allclose(
                    2.0 * env.model.geom_size[env.book_collision_geom_id],
                    [properties.length_m, properties.width_m, properties.thickness_m],
                )
                np.testing.assert_array_equal(
                    env.model.geom_aabb[env.book_collision_geom_id], compiled_aabb
                )
                np.testing.assert_array_equal(env.model.bvh_aabb, compiled_bvh)


class FloatingGripperTest(unittest.TestCase):
    def test_two_pole_force_sensor_step_is_causal_and_monotonic(self) -> None:
        stage1 = np.zeros(1)
        stage2 = np.zeros(1)
        alpha = 1.0 - np.exp(-2.0 * np.pi * 30.0 / 1000.0)
        response = []
        for _ in range(200):
            response.append(
                float(advance_two_pole_filter(stage1, stage2, [1.0], alpha)[0])
            )
        self.assertGreater(response[0], 0.0)
        self.assertLess(response[0], 1.0)
        self.assertTrue(np.all(np.diff(response) >= 0.0))
        self.assertAlmostEqual(response[-1], 1.0, places=8)

        advance_two_pole_filter(stage1, stage2, [0.25], 1.0)
        self.assertEqual(stage2[0], 0.25)

    def test_force_sensor_knob_resolves_timing_and_zero_is_identity(self) -> None:
        with FloatingFlipUpTeleop(seed=0, force_sensor_cutoff_hz=30.0) as env:
            params = env.force_sensor_parameters
            self.assertTrue(params["enabled"])
            self.assertEqual(params["kind"], "cascaded_first_order")
            self.assertAlmostEqual(params["pole_cutoff_hz"], 30.0)
            self.assertAlmostEqual(params["sample_hz"], 1000.0)
            self.assertAlmostEqual(params["step_t50_ms"], 8.904, places=3)
            np.testing.assert_allclose(
                env.sensor_wrench("tool"), env.contact_wrench("tool")
            )

        with FloatingFlipUpTeleop(seed=0, force_sensor_cutoff_hz=0.0) as env:
            self.assertFalse(env.force_sensor_parameters["enabled"])
            env.step(env.tool_home)
            np.testing.assert_array_equal(
                env.sensor_wrench("world"), env.contact_wrench("world")
            )

    def test_force_sensor_rejects_invalid_cutoffs(self) -> None:
        for value in (-0.01, 500.0):
            with self.assertRaisesRegex(ValueError, "force_sensor_cutoff_hz"):
                FloatingFlipUpTeleop(seed=0, force_sensor_cutoff_hz=value)

    def test_tip_softness_interpolates_only_tip_pad_contact(self) -> None:
        with FloatingFlipUpTeleop(seed=0, tip_softness=0.5) as env:
            for side in ("right", "left"):
                tip_id = env.model.geom(f"wsg50/{side}_tip_pad").id
                guard_id = env.model.geom(f"wsg50/{side}_finger_guard").id
                np.testing.assert_allclose(
                    env.model.geom_solref[tip_id], [0.015, 1.5]
                )
                self.assertAlmostEqual(env.model.geom_solimp[tip_id, 2], 0.004)
                np.testing.assert_allclose(
                    env.model.geom_solref[guard_id], [0.010, 1.0]
                )
                self.assertAlmostEqual(env.model.geom_solimp[guard_id, 2], 0.003)
            self.assertEqual(env.tip_contact_parameters["softness"], 0.5)

    def test_tip_softness_rejects_values_outside_unit_interval(self) -> None:
        for value in (-0.01, 1.01):
            with self.assertRaisesRegex(ValueError, "tip_softness"):
                FloatingFlipUpTeleop(seed=0, tip_softness=value)

    def test_floating_gripper_holds_home_without_arm_joints(self) -> None:
        with FloatingFlipUpTeleop(seed=0) as env:
            self.assertEqual(env.controller_kind, "floating_gripper")
            self.assertEqual(env.tool_kp, 5000.0)
            self.assertFalse(any("ur5e" in name for name in env.camera_names()))
            for _ in range(1000):
                env.step(env.tool_home)
            self.assertLess(np.linalg.norm(env.tool_pos - env.tool_home), 5e-4)
            np.testing.assert_array_equal(env.data.qpos[env.gripper_qpos_ids], 0.0)
            np.testing.assert_array_equal(env.data.qvel[env.gripper_dof_ids], 0.0)
            self.assertTrue(np.all(np.isfinite(env.data.qacc)))


if __name__ == "__main__":
    unittest.main()
