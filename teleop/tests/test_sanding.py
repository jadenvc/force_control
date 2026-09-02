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

from sanding_teleop import (  # noqa: E402
    CONTACT_TOOL_Z,
    PANEL_TRANSFORM,
    SandingEnv,
    SandingProperties,
)


def _approach(env, target, steps, max_step_m=0.0005):
    """Slew gradually toward target, like the real CLI driver's --max-speed
    limiter -- stepping the tool straight to a distant target in one tick
    is an unrealistic instantaneous jump that produces a spurious impact
    transient of its own, which is exactly the failure mode this task's
    own break-detection filtering exists to distinguish from real overload.

    Advances an independent commanded waypoint toward ``target``, the same
    way teleop_sanding.py's main loop advances its persistent ``target``
    variable -- NOT by re-deriving the step from the live (still-settling)
    ``env.tool_pos`` each tick, which chases transient tracking overshoot
    instead of making net progress and can walk the wrong direction.
    """
    target = np.asarray(target, dtype=float)
    current = env.tool_pos.copy()
    for _ in range(steps):
        delta = target - current
        norm = np.linalg.norm(delta)
        if norm > max_step_m:
            delta *= max_step_m / norm
        current = current + delta
        env.step(current)


class SandingPropertiesTest(unittest.TestCase):
    def test_rejects_nonpositive_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            SandingProperties(panel_length_m=0.0)
        with self.assertRaises(ValueError):
            SandingProperties(pad_radius_m=-0.01)

    def test_rejects_bad_force_band_ordering(self) -> None:
        with self.assertRaises(ValueError):
            SandingProperties(force_min_n=20.0, force_target_n=12.0)
        with self.assertRaises(ValueError):
            SandingProperties(force_cap_n=10.0, force_break_n=5.0)

    def test_rejects_bad_dose_band_ordering(self) -> None:
        with self.assertRaises(ValueError):
            SandingProperties(dose_low=1.5, dose_high=1.0)

    def test_rejects_out_of_range_pad_softness(self) -> None:
        with self.assertRaises(ValueError):
            SandingProperties(pad_softness=1.5)


class SandingEnvTest(unittest.TestCase):
    def test_free_space_force_is_exactly_zero(self) -> None:
        with SandingEnv() as env:
            self.assertEqual(env.normal_force_n(), 0.0)
            self.assertFalse(env.broken)
            self.assertEqual(env.coverage_fraction("under"), 1.0)

    def test_pressing_into_panel_generates_normal_force_and_dose(self) -> None:
        with SandingEnv() as env:
            hover = np.array([PANEL_TRANSFORM[0, 3], PANEL_TRANSFORM[1, 3], CONTACT_TOOL_Z + 0.05])
            _approach(env, hover, 3000)
            self.assertEqual(env.normal_force_n(), 0.0)

            # NOT tool_kp*penetration=force -- that assumes rigid contact.
            # The default pad_softness=1.0 contact is deliberately soft
            # enough (see SOFT_PAD_SOLREF's comment) that its steady-state
            # stiffness is real and much lower than tool_kp, so the same
            # commanded penetration now produces much less force than that
            # formula predicts. 4mm is a fixed, empirically-checked value
            # that comfortably clears force_target_n at the default contact
            # softness (measured ~25N settled, vs 18N target).
            pressed = hover.copy()
            pressed[2] = CONTACT_TOOL_Z - 0.004
            _approach(env, pressed, 3000)
            for _ in range(2000):
                env.step(pressed)
            self.assertGreater(env.normal_force_n(), 0.0)
            self.assertGreater(env._dose.max(), 0.0)

    def test_dose_rate_law_reaches_low_band_around_target_time(self) -> None:
        # At force_target_n, dose should cross dose_low well before
        # dose_target_time_s (dose reaches 1.0, the *middle* of the band,
        # at dose_target_time_s; the *low* edge of the band is crossed
        # earlier), and should not have crossed it much before that either.
        props = SandingProperties(dose_target_time_s=1.0)
        with SandingEnv(properties=props) as env:
            hover = np.array([PANEL_TRANSFORM[0, 3], PANEL_TRANSFORM[1, 3], CONTACT_TOOL_Z + 0.05])
            _approach(env, hover, 3000)
            # Fixed penetration, not tool_kp*force_target_n -- see
            # test_pressing_into_panel_generates_normal_force_and_dose's
            # comment; the default contact softness's steady-state
            # stiffness is much lower than tool_kp now.
            pressed = hover.copy()
            pressed[2] = CONTACT_TOOL_Z - 0.004
            _approach(env, pressed, 3000)
            # Let contact force actually settle before timing dose
            # accumulation -- this contact softness has a real settling
            # time constant, not just travel time to reach the target.
            for _ in range(2000):
                env.step(pressed)
            steps_to_low_band = None
            for i in range(5000):
                env.step(pressed)
                if env._dose.max() >= props.dose_low:
                    steps_to_low_band = i
                    break
            self.assertIsNotNone(steps_to_low_band)
            # Loose bound: crossing dose_low should happen well within 3x
            # dose_target_time_s, not orders of magnitude off.
            self.assertLess(steps_to_low_band, 3 * props.dose_target_time_s * 1000)

    def test_break_threshold_trips_and_stays_sticky(self) -> None:
        props = SandingProperties(break_force_tau_s=0.0, break_debounce_steps=1)
        with SandingEnv(properties=props) as env:
            hover = np.array([PANEL_TRANSFORM[0, 3], PANEL_TRANSFORM[1, 3], CONTACT_TOOL_Z + 0.05])
            _approach(env, hover, 3000)
            self.assertFalse(env.broken)
            # A big, deliberately unrealistic penetration target that will
            # eventually produce a normal force far above force_break_n --
            # jumping the *target* is intentional here (that's the overload
            # event under test), but the arm still needs several ms of real
            # dynamics to actually travel into that much contact force.
            deep = hover.copy()
            deep[2] = CONTACT_TOOL_Z - 0.01
            # pad_softness=1.0's ~150ms contact time constant means this
            # needs more steps than before to actually ramp up past
            # force_break_n.
            for _ in range(3000):
                env.step(deep)
                if env.broken:
                    break
            self.assertTrue(env.broken)
            # Sticky: pulling back out of contact must not clear it.
            _approach(env, hover, 500)
            self.assertTrue(env.broken)

    def test_success_requires_coverage_and_not_broken(self) -> None:
        with SandingEnv() as env:
            self.assertFalse(env.success())
            env._dose[:] = (env.properties.dose_low + env.properties.dose_high) / 2.0
            self.assertTrue(env.success())
            env._broken = True
            self.assertFalse(env.success())

    def test_reset_zeros_dose_and_clears_broken(self) -> None:
        with SandingEnv() as env:
            env._dose[:] = 1.0
            env._broken = True
            env.reset()
            self.assertTrue(np.all(env._dose == 0.0))
            self.assertFalse(env.broken)
            self.assertTrue(np.all(np.isfinite(env.data.qpos)))

    def test_reset_and_repeated_steps_stay_finite(self) -> None:
        with SandingEnv() as env:
            rng = np.random.default_rng(0)
            for _ in range(200):
                jitter = rng.uniform(-0.02, 0.02, size=3)
                env.step(env.tool_pos + jitter)
            self.assertTrue(np.all(np.isfinite(env.data.qpos)))
            self.assertTrue(np.all(np.isfinite(env._dose)))


class SandingRecorderTest(unittest.TestCase):
    def test_round_trip_records_sanding_specific_fields(self) -> None:
        from sanding_recorder import SandingEpisodeRecorder

        with SandingEnv() as env:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "test.zarr"
                recorder = SandingEpisodeRecorder(path, include_rgb=False, min_samples=5)
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
                    final_coverage_fraction=0.0,
                )
                self.assertIsNotNone(name)

                import zarr

                root = zarr.open(str(path), mode="r")
                self.assertEqual(root.attrs["schema_name"], "pyrite_sanding_sim")
                episode = root["data"][name]
                self.assertEqual(episode["wrench_0"].shape, (30, 6))
                self.assertIn("coverage_just_right", episode.array_keys())
                self.assertIn("broken", episode.array_keys())
                self.assertFalse(bool(episode.attrs["broken"]))

    def test_discards_episode_below_min_samples(self) -> None:
        from sanding_recorder import SandingEpisodeRecorder

        with SandingEnv() as env:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "test.zarr"
                recorder = SandingEpisodeRecorder(path, include_rgb=False, min_samples=20)
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
                name = recorder.commit(
                    success=False,
                    broken=False,
                    termination_reason="test",
                    final_coverage_fraction=0.0,
                )
                self.assertIsNone(name)


if __name__ == "__main__":
    unittest.main()
