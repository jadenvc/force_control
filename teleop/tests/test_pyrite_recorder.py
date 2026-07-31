from __future__ import annotations

import tempfile
import unittest
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import zarr

os.environ.setdefault("MUJOCO_GL", "osmesa")
TELEOP_DIR = Path(__file__).resolve().parents[1]
if str(TELEOP_DIR) not in sys.path:
    sys.path.insert(0, str(TELEOP_DIR))

from flipup_teleop import FlipUpTeleop
from pyrite_recorder import (
    PyriteEpisodeRecorder,
    adaptive_compliance_labels,
    validate_pyrite_dataset,
)


class AdaptiveComplianceLabelTest(unittest.TestCase):
    def test_zero_wrench_keeps_command_and_maximum_stiffness(self) -> None:
        pose = np.array([[0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0]])
        virtual, stiffness = adaptive_compliance_labels(
            pose,
            np.zeros((1, 6)),
            k_max=16000.0,
            k_min=2000.0,
            f_low=2.0,
            f_high=100.0,
        )
        np.testing.assert_allclose(virtual, pose)
        np.testing.assert_allclose(stiffness, [16000.0])

    def test_contact_wrench_offsets_virtual_target_in_tool_frame(self) -> None:
        # The world applies -x force to the robot, so Pyrite's virtual target
        # moves +x into the contact at force / stiffness.
        pose = np.array([[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]])
        wrench = np.array([[-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        virtual, stiffness = adaptive_compliance_labels(
            pose,
            wrench,
            k_max=1000.0,
            k_min=1000.0,
            f_low=1.0,
            f_high=2.0,
        )
        np.testing.assert_allclose(stiffness, [1000.0])
        np.testing.assert_allclose(virtual[0, :3], [0.11, 0.2, 0.3])


class PyriteEpisodeRecorderTest(unittest.TestCase):
    def test_dataset_schema_rate_and_mujoco_state_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "flipup.zarr"
            with FlipUpTeleop(seed=0, settle_s=0.0) as env:
                recorder = PyriteEpisodeRecorder(
                    dataset_path,
                    include_rgb=False,
                    min_samples=2,
                    wrench_filter_seconds=0.0,
                )
                recorder.start_episode({"seed": 0})
                for index in range(2):
                    target = env.tool_pos + np.array([0.0001, 0.0, 0.0])
                    env.step(target, n_substeps=50)
                    recorder.record_sample(
                        env,
                        timestamp_ms=index * 50.0,
                        target_pos=target,
                        target_rotvec=None,
                        device_state=None,
                        sent_force=np.zeros(3),
                        image_rgb=None,
                        image_capture_time_s=None,
                    )
                name = recorder.commit(
                    success=False,
                    termination_reason="test",
                    final_book_angle_deg=env.book_angle_deg(),
                )
                self.assertEqual(name, "episode_0")

                recorder.start_episode({"seed": 0})
                for index in range(2):
                    target = env.tool_pos + np.array([0.0001, 0.0, 0.0])
                    env.step(target, n_substeps=50)
                    recorder.record_sample(
                        env,
                        timestamp_ms=index * 50.0,
                        target_pos=target,
                        target_rotvec=None,
                        device_state=None,
                        sent_force=np.zeros(3),
                        image_rgb=None,
                        image_capture_time_s=None,
                    )
                name = recorder.commit(
                    success=False,
                    termination_reason="test_append",
                    final_book_angle_deg=env.book_angle_deg(),
                )
                self.assertEqual(name, "episode_1")

                summary = validate_pyrite_dataset(dataset_path)
                self.assertEqual(summary["episodes"], 2)
                self.assertEqual(summary["samples"], 4)
                self.assertEqual(summary["sample_hz"], 20.0)

                root = zarr.open(str(dataset_path), mode="r")
                episode = root["data"]["episode_0"]
                self.assertEqual(episode["wrench_0"].shape, (2, 6))
                self.assertEqual(
                    episode["wrench_ground_truth_0"].shape,
                    (2, 6),
                )
                self.assertEqual(episode["ts_pose_command_0"].shape, (2, 7))
                self.assertEqual(
                    episode["ts_pose_virtual_target_0"].shape,
                    (2, 7),
                )
                self.assertEqual(episode["stiffness_0"].shape, (2,))

                state = np.asarray(episode["mujoco_state"][0])
                spec = int(episode.attrs["mujoco_state_spec"])
                mujoco.mj_setState(env.model.ptr, env.data.ptr, state, spec)
                mujoco.mj_forward(env.model.ptr, env.data.ptr)
                np.testing.assert_allclose(env.data.qpos, episode["qpos"][0])


if __name__ == "__main__":
    unittest.main()
