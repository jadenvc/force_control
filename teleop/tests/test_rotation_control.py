from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial.transform import Rotation

os.environ.setdefault("MUJOCO_GL", "osmesa")
TELEOP_DIR = Path(__file__).resolve().parents[1]
if str(TELEOP_DIR) not in sys.path:
    sys.path.insert(0, str(TELEOP_DIR))

import fd_omega  # noqa: E402
from fd_omega import FDOmega  # noqa: E402
from flipup_teleop import FlipUpTeleop  # noqa: E402
from teleop_flipup import build_pos_map, map_wrist_orientation  # noqa: E402


def _tool_rotation(environment: FlipUpTeleop) -> Rotation:
    quaternion_wxyz = environment.tool_quat
    return Rotation.from_quat(quaternion_wxyz[[1, 2, 3, 0]])


class WristInputTest(unittest.TestCase):
    def test_rotation_matrix_validation(self) -> None:
        valid = Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix()
        self.assertTrue(FDOmega._is_rotation_matrix(valid))
        self.assertFalse(FDOmega._is_rotation_matrix(np.zeros((3, 3))))
        self.assertFalse(FDOmega._is_rotation_matrix(np.full((3, 3), np.nan)))

    def test_world_frame_wrist_delta_reaches_tool_command(self) -> None:
        rotation_map = build_pos_map("-x,-y,z")
        device_home = Rotation.from_euler("xyz", [0.3, -0.2, 0.1]).as_matrix()
        device_delta = Rotation.from_rotvec([0.0, 0.0, np.deg2rad(20.0)])
        device_now = device_delta.as_matrix() @ device_home
        tool_home = Rotation.from_euler("xyz", [0.0, -0.5, 0.2])

        command_rotvec, delta_rotvec = map_wrist_orientation(
            device_now,
            device_home,
            rotation_map,
            tool_home.as_rotvec(),
            frame="world",
        )

        np.testing.assert_allclose(
            delta_rotvec,
            [0.0, 0.0, np.deg2rad(20.0)],
            atol=1e-10,
        )
        actual_delta = Rotation.from_rotvec(command_rotvec) * tool_home.inv()
        np.testing.assert_allclose(
            actual_delta.as_rotvec(),
            delta_rotvec,
            atol=1e-10,
        )

    def test_device_open_seeds_wrist_frame_before_background_thread(self) -> None:
        initial_rotation = Rotation.from_euler(
            "xyz", [0.3, -0.2, 0.1]
        ).as_matrix()
        fake_thread = MagicMock()
        sdk_replacements = {
            "Init": MagicMock(),
            "Open": MagicMock(return_value=0),
            "GetSystemName": MagicMock(return_value="mock omega.7"),
            "GetSerialNumber": MagicMock(return_value=123),
            "HasWrist": MagicMock(return_value=True),
            "HasActiveGripper": MagicMock(return_value=False),
            "IsInitialized": MagicMock(return_value=True),
            "Start": MagicMock(return_value=0),
            "Stop": MagicMock(return_value=0),
            "EnableForce": MagicMock(return_value=0),
            "GetPosition": MagicMock(return_value=(0, 0.0, 0.0, 0.0)),
            "GetOrientationFrame": MagicMock(
                return_value=(0, initial_rotation)
            ),
            "SetForce": MagicMock(return_value=0),
            "SetBrakes": MagicMock(return_value=0),
            "Close": MagicMock(return_value=0),
        }
        with (
            patch.multiple(fd_omega.fdsdk, **sdk_replacements),
            patch.object(fd_omega.threading, "Thread", return_value=fake_thread),
            patch.object(fd_omega.time, "sleep"),
        ):
            device = FDOmega(read_orientation=True)
            device.open()
            state = device.get_state()
            device.close()

        self.assertTrue(state["has_wrist"])
        self.assertTrue(state["orientation_valid"])
        self.assertEqual(state["orientation_sample_count"], 1)
        np.testing.assert_allclose(state["rot"], initial_rotation)
        fake_thread.start.assert_called_once()


class RotationalImpedanceTest(unittest.TestCase):
    def test_default_gains_track_slew_without_overshoot_or_saturation(self) -> None:
        with FlipUpTeleop(seed=0, settle_s=2.5, offscreen=(64, 64)) as environment:
            actuator_limits = np.asarray(environment.model.actuator_forcerange)[
                environment.actuator_ids, 1
            ]
            for axis in np.eye(3):
                environment.reset()
                target_position = environment.tool_pos.copy()
                home = _tool_rotation(environment)
                projected_angles = []
                errors = []
                saturated = []

                ramp_steps = 167  # 10 degrees at the default 60 degrees/second
                for step in range(1000):
                    fraction = min(1.0, (step + 1) / ramp_steps)
                    delta = Rotation.from_rotvec(
                        axis * np.deg2rad(10.0) * fraction
                    )
                    command = delta * home
                    environment.step(
                        target_position,
                        target_rotvec=command.as_rotvec(),
                    )

                    actual = _tool_rotation(environment)
                    projected_angles.append(
                        np.degrees(
                            np.dot((actual * home.inv()).as_rotvec(), axis)
                        )
                    )
                    errors.append(
                        np.degrees((command * actual.inv()).magnitude())
                    )
                    controls = np.abs(
                        environment.data.ctrl[environment.actuator_ids]
                    )
                    saturated.append(
                        bool(np.any(controls >= actuator_limits - 0.05))
                    )

                self.assertLessEqual(max(projected_angles), 10.05)
                self.assertLess(errors[-1], 0.5)
                self.assertFalse(any(saturated))


if __name__ == "__main__":
    unittest.main()
