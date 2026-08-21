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

from fd_omega import (  # noqa: E402
    FDOmega,
    advance_grip_force,
    advance_reflected_force,
)
from teleop_flipup import (  # noqa: E402
    GripOpenCalibrator,
    collection_home_metrics,
    episode_start_safety,
    smooth_collection_force_gain,
)


class CollectionSafetyTest(unittest.TestCase):
    def test_relaxed_grip_opening_auto_calibrates_below_nominal_limit(self) -> None:
        calibration = GripOpenCalibrator(
            closed_m=0.0,
            nominal_open_m=0.025,
            stable_s=0.35,
            tolerance_m=0.002,
            minimum_span_m=0.006,
        )
        self.assertFalse(calibration.observe(0.016, 0.0))
        self.assertFalse(calibration.observe(0.0161, 0.20))
        self.assertTrue(calibration.observe(0.0161, 0.36))
        self.assertAlmostEqual(calibration.effective_open_m, 0.0161)
        self.assertFalse(calibration.observe(0.012, 0.50))

    def test_negative_omega_gap_is_inferred_as_open_direction(self) -> None:
        calibration = GripOpenCalibrator(
            closed_m=0.0,
            nominal_open_m=0.025,
            stable_s=0.20,
            tolerance_m=0.002,
            minimum_span_m=0.006,
        )
        self.assertFalse(calibration.observe(-0.0262, 0.0))
        self.assertTrue(calibration.observe(-0.0262, 0.21))
        self.assertAlmostEqual(calibration.effective_open_m, -0.025)
        self.assertFalse(calibration.observe(-0.020, 0.30))
        self.assertTrue(calibration.observe(-0.024, 0.30))

    def test_grip_calibration_rejects_nearly_closed_false_endpoint(self) -> None:
        calibration = GripOpenCalibrator(
            closed_m=0.0,
            nominal_open_m=0.025,
            stable_s=0.1,
            minimum_span_m=0.006,
        )
        calibration.observe(0.003, 0.0)
        self.assertFalse(calibration.observe(0.003, 1.0))
        self.assertIsNone(calibration.reference_m)

    def test_grip_calibration_waits_until_slow_opening_stops(self) -> None:
        calibration = GripOpenCalibrator(
            closed_m=0.0,
            nominal_open_m=0.025,
            stable_s=0.20,
            minimum_span_m=0.006,
        )
        for index in range(40):
            ready = calibration.observe(0.010 + 0.0001 * index, 0.01 * index)
        self.assertFalse(ready)
        self.assertFalse(calibration.observe(0.014, 0.50))
        self.assertTrue(calibration.observe(0.014, 0.61))

    def test_grip_feedback_filter_obeys_rate_limit_and_converges(self) -> None:
        filtered = applied = 0.0
        history = []
        for _ in range(1000):
            filtered, applied = advance_grip_force(
                filtered,
                applied,
                -3.0,
                dt_s=0.001,
                tau_s=0.010,
                rate_n_s=60.0,
            )
            history.append(applied)
        delta = np.diff(np.r_[0.0, history])
        self.assertTrue(np.all(np.abs(delta) <= 0.0600001))
        self.assertAlmostEqual(history[-1], -3.0, places=8)

    def test_home_metrics_use_fixed_home_and_measured_velocity(self) -> None:
        distance, speed, valid = collection_home_metrics(
            {
                "pos": np.array([0.022, 0.0, -0.02]),
                "vel": np.array([0.0, 0.006, 0.008]),
                "velocity_valid": True,
            },
            np.array([0.02, 0.0, -0.02]),
        )
        self.assertAlmostEqual(distance, 0.002)
        self.assertAlmostEqual(speed, 0.010)
        self.assertTrue(valid)

    def test_collection_force_gain_holds_then_ramps_smoothly(self) -> None:
        self.assertEqual(smooth_collection_force_gain(0.05, 0.1, 0.4), 0.0)
        self.assertEqual(smooth_collection_force_gain(0.10, 0.1, 0.4), 0.0)
        self.assertAlmostEqual(
            smooth_collection_force_gain(0.30, 0.1, 0.4), 0.5
        )
        self.assertEqual(smooth_collection_force_gain(0.50, 0.1, 0.4), 1.0)
        self.assertEqual(smooth_collection_force_gain(2.00, 0.1, 0.4), 1.0)

    def test_contact_heavy_or_unsettled_starts_are_rejected(self) -> None:
        self.assertTrue(
            episode_start_safety(
                0.0003,
                0.0,
                max_settle_error_m=0.01,
                max_contact_force_n=0.5,
            )
        )
        self.assertFalse(
            episode_start_safety(
                0.002,
                29.6,
                max_settle_error_m=0.01,
                max_contact_force_n=0.5,
            )
        )
        self.assertFalse(
            episode_start_safety(
                0.02,
                0.0,
                max_settle_error_m=0.01,
                max_contact_force_n=0.5,
            )
        )

    def test_hard_force_reset_discards_device_filter_history(self) -> None:
        device = FDOmega()
        device._reflected = np.array([1.0, 2.0, 3.0])
        device._reflected_applied = np.array([0.5, 1.0, 1.5])
        device.clear_reflected_force()
        np.testing.assert_array_equal(device._reflected, np.zeros(3))
        np.testing.assert_array_equal(device._reflected_applied, np.zeros(3))

    def test_idle_centering_can_be_switched_off_before_recording(self) -> None:
        device = FDOmega(spring_k=100.0, spring_max_force=2.0)
        self.assertTrue(device.get_state()["centering_enabled"])
        device.set_centering_enabled(False)
        self.assertFalse(device.get_state()["centering_enabled"])
        device.set_centering_enabled(True)
        self.assertTrue(device.get_state()["centering_enabled"])

    def test_centering_force_cap_cannot_be_negative(self) -> None:
        with self.assertRaises(ValueError):
            FDOmega(spring_k=100.0, spring_max_force=-1.0)

    def test_device_force_slew_depends_on_elapsed_time_not_callback_count(self) -> None:
        def advance(dt, count):
            filtered = np.zeros(3)
            applied = np.zeros(3)
            for _ in range(count):
                filtered, applied = advance_reflected_force(
                    filtered,
                    applied,
                    np.array([10.0, 0.0, 0.0]),
                    dt_s=dt,
                    tau_s=0.0,
                    rate_n_s=120.0,
                )
            return applied

        np.testing.assert_allclose(advance(0.001, 10), [1.2, 0.0, 0.0])
        np.testing.assert_allclose(advance(0.005, 2), [1.2, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
