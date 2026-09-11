from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
TELEOP_DIR = Path(__file__).resolve().parents[1]
if str(TELEOP_DIR) not in sys.path:
    sys.path.insert(0, str(TELEOP_DIR))

from insertion_teleop import (  # noqa: E402
    CONTACT_CONTROL_Z,
    DynamicFilter,
    EMAFilter,
    InsertionEnv,
    InsertionProperties,
)
from insertion_scripted_demo import ScriptedDemoConfig, run_scripted_demo  # noqa: E402

# Documented sane ceiling for this task -- see insertion_teleop.py's
# InsertionProperties.force_break_n comment for why 45N (not, e.g., 40N) is
# the chosen break threshold; the scripted demo's own spiral search is
# designed to stay comfortably under it (see
# insertion_scripted_demo.ScriptedDemoConfig.search_spiral_max_radius_m).
FORCE_CEILING_N = 45.0


def _approach(env, target, steps, max_step_m=0.0005):
    """Slew gradually toward target -- same rationale/pattern as
    test_sanding.py's ``_approach`` helper: an instantaneous jump to a
    distant target produces its own spurious impact transient, exactly what
    this task's break-detection filtering is trying to distinguish real
    overload from."""
    target = np.asarray(target, dtype=float)
    current = env.tool_pos.copy()
    for _ in range(steps):
        delta = target - current
        norm = np.linalg.norm(delta)
        if norm > max_step_m:
            delta *= max_step_m / norm
        current = current + delta
        env.step(current)


class InsertionPropertiesTest(unittest.TestCase):
    def test_rejects_nonpositive_insert_depth(self) -> None:
        with self.assertRaises(ValueError):
            InsertionProperties(insert_depth_target_m=0.0)

    def test_rejects_insert_depth_past_socket_floor(self) -> None:
        with self.assertRaises(ValueError):
            InsertionProperties(insert_depth_target_m=0.2)  # deeper than the 0.05m socket

    def test_rejects_bad_force_threshold_ordering(self) -> None:
        with self.assertRaises(ValueError):
            InsertionProperties(force_contact_threshold_n=50.0, force_break_n=10.0)

    def test_rejects_out_of_range_peg_softness(self) -> None:
        with self.assertRaises(ValueError):
            InsertionProperties(peg_softness=1.5)

    def test_rejects_bad_ft_filter_type(self) -> None:
        with self.assertRaises(ValueError):
            InsertionProperties(ft_filter_type="bogus")

    def test_rejects_negative_wiggle_amplitude(self) -> None:
        with self.assertRaises(ValueError):
            InsertionProperties(search_wiggle_amplitude_n=-1.0)

    def test_rejects_negative_hole_tilt_randomization(self) -> None:
        with self.assertRaises(ValueError):
            InsertionProperties(hole_tilt_randomization_deg=-1.0)


class DynamicFilterTest(unittest.TestCase):
    def test_converges_toward_constant_input(self) -> None:
        f = DynamicFilter(alpha=0.9, beta=0.3, dt=0.001)
        target = np.array([1.0, -2.0, 3.0, 0.0, 0.0, 0.0])
        out = np.zeros(6)
        for _ in range(20000):
            out = f.step(target, 0.001)
        np.testing.assert_allclose(out, target, atol=1e-2)

    def test_reset_zeros_state(self) -> None:
        f = DynamicFilter()
        f.step(np.ones(6), 0.001)
        f.reset()
        np.testing.assert_array_equal(f.F_ff, np.zeros(6))
        np.testing.assert_array_equal(f.F_ff_dot, np.zeros(6))


class EMAFilterTest(unittest.TestCase):
    def test_first_call_returns_input_unchanged(self) -> None:
        f = EMAFilter(alpha=0.2)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(f(x), x)

    def test_smooths_toward_constant_input(self) -> None:
        f = EMAFilter(alpha=0.2)
        x = np.array([5.0, 5.0, 5.0])
        out = None
        for _ in range(200):
            out = f(x)
        np.testing.assert_allclose(out, x, atol=1e-6)


class InsertionEnvTest(unittest.TestCase):
    def test_free_space_force_is_exactly_zero(self) -> None:
        with InsertionEnv() as env:
            self.assertEqual(env.normal_force_n(), 0.0)
            self.assertFalse(env.broken)
            self.assertFalse(env.success())

    def test_reset_and_repeated_steps_stay_finite(self) -> None:
        with InsertionEnv() as env:
            rng = np.random.default_rng(0)
            for _ in range(500):
                jitter = rng.uniform(-0.005, 0.005, size=3)
                ok = env.step(env.tool_pos + jitter)
                self.assertTrue(ok)
            self.assertTrue(np.all(np.isfinite(env.data.qpos)))
            self.assertTrue(np.all(np.isfinite(env.data.qvel)))

    def test_task_space_step_with_feed_forward_does_not_nan(self) -> None:
        with InsertionEnv() as env:
            target = env.tool_pos.copy()
            ff = np.array([1.0, -1.0, -4.0, 0.0, 0.0, 0.0])
            for _ in range(2000):
                ok = env.step(target, feed_forward_wrench=ff)
                self.assertTrue(ok)
            self.assertTrue(np.all(np.isfinite(env.data.qpos)))
            self.assertTrue(np.all(np.isfinite(env.data.qvel)))

    def test_pressing_against_fixture_top_generates_bounded_force(self) -> None:
        """Landing off-center (missing the opening) should register contact
        against the fixture's top lip with a bounded, non-runaway force --
        this is the exact scenario insertion_teleop.py's tuning log (see
        DEFAULT_TOOL_KP's comment) found originally produced 100-250N spikes
        before tool_kp was lowered."""
        with InsertionEnv() as env:
            entrance = env.hole_entrance_pos.copy()
            hover = entrance.copy()
            hover[0] += 0.02  # off-center: over the frame, not the opening
            hover[2] = CONTACT_CONTROL_Z + 0.03
            _approach(env, hover, 3000)
            self.assertEqual(env.normal_force_n(), 0.0)

            pressed = hover.copy()
            pressed[2] = CONTACT_CONTROL_Z - 0.0035
            _approach(env, pressed, 3000)
            for _ in range(1000):
                env.step(pressed, feed_forward_wrench=np.array([0, 0, -4.0, 0, 0, 0]))
            force = env.normal_force_n()
            self.assertGreater(force, 0.0)
            self.assertLess(force, FORCE_CEILING_N)

    def test_success_requires_depth_hold_and_not_broken(self) -> None:
        with InsertionEnv() as env:
            self.assertFalse(env.success())
            env._depth_hold_steps = env.properties.success_hold_steps
            self.assertTrue(env.success())
            env._broken = True
            self.assertFalse(env.success())

    def test_reset_clears_broken_and_depth_hold(self) -> None:
        with InsertionEnv() as env:
            env._broken = True
            env._depth_hold_steps = 999
            env.reset()
            self.assertFalse(env.broken)
            self.assertEqual(env._depth_hold_steps, 0)
            self.assertTrue(np.all(np.isfinite(env.data.qpos)))

    def test_wrist_wrench_filtered_matches_raw_shape(self) -> None:
        with InsertionEnv() as env:
            raw = env.wrist_wrench_raw()
            filtered = env.wrist_wrench_filtered()
            self.assertEqual(raw.shape, (6,))
            self.assertEqual(filtered.shape, (6,))


class ScriptedDemoTest(unittest.TestCase):
    def test_scripted_demo_completes_and_inserts_the_peg(self) -> None:
        with InsertionEnv(seed=0) as env:
            result = run_scripted_demo(env=env, seed=0, max_steps=40000)
            self.assertTrue(result.success, msg=f"termination_reason={result.termination_reason}")
            self.assertGreaterEqual(env.peg_tip_depth_m(), env.properties.insert_depth_target_m)

    def test_scripted_demo_force_never_exceeds_documented_ceiling(self) -> None:
        with InsertionEnv(seed=1) as env:
            result = run_scripted_demo(env=env, seed=1, max_steps=40000)
            self.assertLessEqual(result.peak_force_n, FORCE_CEILING_N)

    def test_scripted_demo_mostly_succeeds_across_seeds(self) -> None:
        # A handful of seeds, not an exhaustive sweep (keep the test fast) --
        # see README_insertion.md's tuning log for the full 20-seed sweep
        # (19/20 success) this is a smaller sample of.
        successes = 0
        for seed in range(5):
            with InsertionEnv(seed=seed) as env:
                result = run_scripted_demo(env=env, seed=seed, max_steps=40000)
                successes += int(result.success)
        self.assertGreaterEqual(successes, 4)


class HoleTiltRandomizationTest(unittest.TestCase):
    """hole_tilt_randomization_deg -- mirrors peg_tilt_randomization_deg's
    own no-op/range tests, plus a check that hole_entrance's world position
    stays fixed (the socket tilts IN PLACE, walls swing around it, see
    InsertionEnv.hole_body_id's comment)."""

    def test_zero_is_a_true_noop(self) -> None:
        with InsertionEnv(properties=InsertionProperties(hole_tilt_randomization_deg=0.0)) as env:
            baseline_R = env.hole_rotation_matrix.copy()
            baseline_pos = env.hole_entrance_pos.copy()
            for _ in range(5):
                env.reset()
                np.testing.assert_array_equal(env.hole_rotation_matrix, np.eye(3))
                np.testing.assert_array_equal(env.hole_rotation_matrix, baseline_R)
                np.testing.assert_allclose(env.hole_entrance_pos, baseline_pos, atol=0.0)

    def test_nonzero_samples_within_range_and_keeps_entrance_fixed(self) -> None:
        max_deg = 15.0
        with InsertionEnv(
            seed=0, properties=InsertionProperties(hole_tilt_randomization_deg=max_deg)
        ) as env:
            entrance0 = env.hole_entrance_pos.copy()
            saw_nonidentity = False
            for _ in range(10):
                env.reset()
                R = env.hole_rotation_matrix
                # hole_entrance's WORLD position must stay exactly fixed
                # regardless of tilt -- rotating the socket body about its
                # own (0,0,0)-local-pos origin, which is where the site
                # sits, must not move that origin.
                np.testing.assert_allclose(env.hole_entrance_pos, entrance0, atol=1e-9)
                # R must be a valid rotation (orthonormal, det=+1).
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
                self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)
                bore_axis = R @ np.array([0.0, 0.0, 1.0])
                angle_from_world_z = np.degrees(
                    np.arccos(np.clip(bore_axis[2], -1.0, 1.0))
                )
                # Composed roll+pitch (each up to max_deg) can tilt the bore
                # axis by more than max_deg alone; generous bound.
                self.assertLessEqual(angle_from_world_z, 2.0 * max_deg + 1e-6)
                if not np.allclose(R, np.eye(3)):
                    saw_nonidentity = True
            self.assertTrue(saw_nonidentity, "expected at least one tilted episode across 10 resets")

    def test_peg_tip_depth_uses_tilted_hole_frame_not_world_z(self) -> None:
        """peg_tip_depth_m() must project along the hole's ACTUAL (tilted)
        bore axis, not a hardcoded world -z. Tests the FORMULA directly
        (writes the tool site's xpos, bypassing the arm's dynamics/
        controller entirely) rather than trying to physically drive a
        STRAIGHT peg down a substantially TILTED bore -- that's a separate,
        real geometric problem (a straight peg can legitimately catch the
        tilted rim before reaching an arbitrary interior point, exactly the
        kind of tilt-vs-orientation mismatch --peg-tilt-randomization-deg's
        own scripted-demo caveat already documents for the peg side), not a
        depth-FORMULA bug -- isolating the formula from that confound is
        the honest way to check this specific claim."""
        with InsertionEnv(
            seed=0, properties=InsertionProperties(hole_tilt_randomization_deg=20.0)
        ) as env:
            env.reset()
            R = env.hole_rotation_matrix
            if np.allclose(R, np.eye(3)):
                # Extremely unlikely (continuous uniform sample landing at
                # exactly 0,0) but keep the test deterministic if it happens.
                self.skipTest("sampled hole tilt was exactly zero")
            local_bore_down = np.array([0.0, 0.0, -1.0])
            world_bore_down = R @ local_bore_down
            depth_m = 0.012
            tip_target = env.hole_entrance_pos + depth_m * world_bore_down
            env.data.site(env.tool_site_id).xpos[:] = tip_target
            measured = env.peg_tip_depth_m()
            self.assertAlmostEqual(measured, depth_m, places=9)

            # And the reverse direction (above the fixture, along the same
            # tilted axis) must read negative, same convention as untilted.
            above_target = env.hole_entrance_pos - depth_m * world_bore_down
            env.data.site(env.tool_site_id).xpos[:] = above_target
            self.assertAlmostEqual(env.peg_tip_depth_m(), -depth_m, places=9)


class ToolKpAxesHoleFrameAlignmentTest(unittest.TestCase):
    """The actual motivating question: does --tool-kp-axes' stiffness
    diagonal track a tilted HOLE's real bore axis, or does it stay pinned
    to world -z? Validates the effective world-frame stiffness matrix's
    principal axis against the hole's current bore direction directly,
    with a numeric angle, rather than merely asserting no crash."""

    def test_principal_stiffness_axis_tracks_tilted_hole_bore_axis(self) -> None:
        props = InsertionProperties(hole_tilt_randomization_deg=15.0)
        with InsertionEnv(
            seed=0, properties=props, tool_kp=1200.0, tool_kp_axes=(0.5, 0.5, 2.0)
        ) as env:
            env.reset()
            R = env.hole_rotation_matrix
            self.assertFalse(np.allclose(R, np.eye(3)), "expected a nonzero sampled hole tilt")

            K = env.task_space_kp[:3, :3]
            # K must be symmetric (a valid rotated stiffness tensor).
            np.testing.assert_allclose(K, K.T, atol=1e-9)
            eigvals, eigvecs = np.linalg.eigh(K)
            principal_axis = eigvecs[:, int(np.argmax(eigvals))]
            # tool_kp_axes' largest component (2.0, index 2 -- "Z"/bore) is
            # unique and strictly bigger than X/Y (0.5 each), so the
            # top eigenvalue must correspond to the (rotated) hole-frame Z
            # axis, with no degeneracy to worry about.
            hole_bore_axis_world = R @ np.array([0.0, 0.0, 1.0])
            if np.dot(principal_axis, hole_bore_axis_world) < 0.0:
                principal_axis = -principal_axis
            cos_angle = np.clip(np.dot(principal_axis, hole_bore_axis_world), -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cos_angle)))
            # The number this whole feature is FOR: aligned with the hole's
            # actual bore axis to numerical precision, not off by anything
            # close to the sampled tilt magnitude (~15 degrees) the way an
            # un-rotated world-frame diagonal would be.
            self.assertLess(angle_deg, 1e-6)

            # Sanity check the eigenvalues themselves match tool_kp*axes.
            np.testing.assert_allclose(sorted(eigvals), sorted([600.0, 600.0, 2400.0]), atol=1e-6)

    def test_zero_hole_tilt_reduces_to_original_world_frame_diagonal(self) -> None:
        """No-op check: at hole_tilt_randomization_deg=0, the rotated-frame
        formula must be BYTE-IDENTICAL to the original bare world-frame
        diagonal (R is exactly np.eye(3)), not merely close."""
        with InsertionEnv(tool_kp=1200.0, tool_kp_axes=(0.5, 0.5, 2.0)) as env:
            np.testing.assert_array_equal(env.hole_rotation_matrix, np.eye(3))
            expected = np.diag([600.0, 600.0, 2400.0])
            np.testing.assert_array_equal(env.task_space_kp[:3, :3], expected)

    def test_cartesian_damping_also_rotates_with_hole_frame(self) -> None:
        props = InsertionProperties(hole_tilt_randomization_deg=15.0, cartesian_damping_scale=1.0)
        with InsertionEnv(
            seed=1, properties=props, tool_kp=1200.0, tool_kp_axes=(0.5, 0.5, 2.0)
        ) as env:
            env.reset()
            R = env.hole_rotation_matrix
            self.assertFalse(np.allclose(R, np.eye(3)))
            Kd = env.task_space_cartesian_kd[:3, :3]
            np.testing.assert_allclose(Kd, Kd.T, atol=1e-9)
            eigvals, eigvecs = np.linalg.eigh(Kd)
            principal_axis = eigvecs[:, int(np.argmax(eigvals))]
            hole_bore_axis_world = R @ np.array([0.0, 0.0, 1.0])
            if np.dot(principal_axis, hole_bore_axis_world) < 0.0:
                principal_axis = -principal_axis
            cos_angle = np.clip(np.dot(principal_axis, hole_bore_axis_world), -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cos_angle)))
            self.assertLess(angle_deg, 1e-6)

    def test_step_task_space_still_works_with_nondiagonal_kp_kd(self) -> None:
        """step_task_space's task_space_kp/task_space_cartesian_kd usage
        must be a general matrix-vector product, not assuming diagonal
        structure -- exercise it directly at a nonzero hole tilt and check
        the sim stays finite/stable."""
        props = InsertionProperties(hole_tilt_randomization_deg=15.0)
        with InsertionEnv(seed=0, properties=props, tool_kp=1200.0, tool_kp_axes=(0.5, 0.5, 2.0)) as env:
            env.reset()
            self.assertFalse(np.allclose(env.task_space_kp[:3, :3], np.diag(np.diag(env.task_space_kp[:3, :3]))))
            target = env.tool_pos.copy()
            for _ in range(2000):
                ok = env.step(target)
                self.assertTrue(ok)
            self.assertTrue(np.all(np.isfinite(env.data.qpos)))
            self.assertTrue(np.all(np.isfinite(env.data.qvel)))


class InsertionRecorderTest(unittest.TestCase):
    def test_round_trip_records_insertion_specific_fields(self) -> None:
        from insertion_recorder import InsertionEpisodeRecorder

        with InsertionEnv() as env:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "test.zarr"
                recorder = InsertionEpisodeRecorder(path, include_rgb=False, min_samples=5)
                recorder.start_episode(metadata={"note": "unit test"})
                target = env.tool_pos.copy()
                for i in range(30):
                    env.step(target)
                    recorder.record_sample(
                        env,
                        timestamp_ms=float(i),
                        target_pos=target,
                        target_rotvec=None,
                        device_state={"pos": np.zeros(3), "vel": np.zeros(3)},
                        sent_force=np.zeros(3),
                        image_rgb=None,
                    )
                name = recorder.commit(
                    success=False,
                    broken=False,
                    termination_reason="test",
                )
                self.assertIsNotNone(name)

                import zarr

                root = zarr.open(str(path), mode="r")
                self.assertEqual(root.attrs["schema_name"], "pyrite_insertion_sim")
                episode = root["data"][name]
                self.assertEqual(episode["wrench_0"].shape, (30, 6))
                self.assertIn("peg_tip_depth_m", episode.array_keys())
                self.assertIn("wrench_filtered_0", episode.array_keys())
                self.assertFalse(bool(episode.attrs["broken"]))

    def test_discards_episode_below_min_samples(self) -> None:
        from insertion_recorder import InsertionEpisodeRecorder

        with InsertionEnv() as env:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "test.zarr"
                recorder = InsertionEpisodeRecorder(path, include_rgb=False, min_samples=20)
                recorder.start_episode()
                target = env.tool_pos.copy()
                for i in range(3):
                    env.step(target)
                    recorder.record_sample(
                        env,
                        timestamp_ms=float(i),
                        target_pos=target,
                        target_rotvec=None,
                        device_state=None,
                        sent_force=np.zeros(3),
                        image_rgb=None,
                    )
                name = recorder.commit(success=False, broken=False, termination_reason="test")
                self.assertIsNone(name)


if __name__ == "__main__":
    unittest.main()
