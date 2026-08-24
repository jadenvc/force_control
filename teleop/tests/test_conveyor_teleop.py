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

from conveyor_teleop import ConveyorTeleop, predicted_feel  # noqa: E402
from pyrite_recorder import (  # noqa: E402
    CONVEYOR_SCHEMA_NAME,
    PyriteEpisodeRecorder,
    validate_pyrite_dataset,
)


class ConveyorTeleopInterfaceTest(unittest.TestCase):
    """The pieces teleop_ball/teleop_flipup's machinery expects from an env."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = ConveyorTeleop(
            seed=0, episode_index=0, belt_speed_m_per_s=0.15, respawn_object=False
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_reset_settles_at_the_home_pose_with_the_grasp_orientation(self) -> None:
        env = self.env
        env.reset(episode_index=0)
        self.assertLess(float(np.linalg.norm(env.tool_pos - env.tool_home)), 1e-3)
        # 180 degrees about world x: fingers straight down, closing across x.
        np.testing.assert_allclose(env.home_rotvec, [np.pi, 0.0, 0.0], atol=1e-9)
        self.assertLess(
            float(np.linalg.norm(env.tool_quat - np.array([0.0, 1.0, 0.0, 0.0]))),
            2e-3,
        )

    def test_target_pose7_uses_the_fixed_grasp_orientation(self) -> None:
        pose = self.env.target_pose7([0.2, 0.3, 0.4])
        np.testing.assert_allclose(pose[:3], [0.2, 0.3, 0.4])
        np.testing.assert_allclose(pose[3:], [0.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(float(np.linalg.norm(pose[3:])), 1.0)

    def test_target_pose7_accepts_an_operator_rotvec(self) -> None:
        pose = self.env.target_pose7([0.2, 0.3, 0.4], target_rotvec=[np.pi, 0.0, 0.0])
        self.assertAlmostEqual(abs(float(np.dot(pose[3:], [0.0, 1.0, 0.0, 0.0]))), 1.0)

    def test_limited_target_is_a_no_op_when_the_force_limit_is_off(self) -> None:
        env = self.env
        self.assertEqual(env.tool_force_limit, 0.0)
        target = env.tool_pos + np.array([0.05, 0.0, 0.0])
        np.testing.assert_allclose(env.limited_target(target), target)

    def test_limited_target_saturates_the_commanded_force(self) -> None:
        env = ConveyorTeleop(
            seed=0, belt_speed_m_per_s=0.0, tool_force_limit=20.0, respawn_object=False
        )
        try:
            target = env.tool_pos + np.array([1.0, 0.0, 0.0])
            limited = env.limited_target(target)
            commanded = env.tool_kp * float(np.linalg.norm(limited - env.tool_pos))
            self.assertLessEqual(commanded, 20.0 + 1e-6)
            # Direction is preserved; only the magnitude is squashed.
            direction = (limited - env.tool_pos) / np.linalg.norm(limited - env.tool_pos)
            np.testing.assert_allclose(direction, [1.0, 0.0, 0.0], atol=1e-9)
        finally:
            env.close()

    def test_contact_force_is_exactly_zero_in_free_space(self) -> None:
        env = self.env
        env.reset(episode_index=0)
        target = env.tool_pos.copy()
        for _ in range(600):
            target[1] -= 0.10 * env.timestep
            env.step(target, gripper_width=env.OPEN_GRIPPER_WIDTH_M)
            np.testing.assert_array_equal(env.contact_force(), np.zeros(3))
        self.assertEqual(env.grip_force(), 0.0)

    def test_estimated_force_opposes_the_commanded_motion(self) -> None:
        env = self.env
        env.reset(episode_index=0)
        target = env.tool_pos + np.array([0.0, -0.01, 0.0])
        for _ in range(200):
            env.step(target, gripper_width=env.OPEN_GRIPPER_WIDTH_M)
        estimated = env.estimated_force(target)
        # The tool is being pulled toward -y, so the reflected force is +y.
        self.assertGreater(estimated[1], 0.0)

    def test_reflected_force_sources(self) -> None:
        env = self.env
        env.reset(episode_index=0)
        target = env.tool_pos.copy()
        np.testing.assert_array_equal(env.reflected_force("none", target), np.zeros(3))
        for source in ("contact", "wrist", "estimated"):
            force = env.reflected_force(source, target)
            self.assertEqual(force.shape, (3,))
            self.assertLessEqual(float(np.linalg.norm(force)), env.force_clip + 1e-6)
        with self.assertRaises(ValueError):
            env.reflected_force("nonsense", target)

    def test_wrench_frames_agree_in_magnitude(self) -> None:
        env = self.env
        for method in (env.contact_wrench, env.wrist_wrench):
            world = method(frame="world")
            tool = method(frame="tool")
            np.testing.assert_allclose(
                np.linalg.norm(world[:3]), np.linalg.norm(tool[:3]), atol=1e-9
            )
            np.testing.assert_allclose(
                np.linalg.norm(world[3:]), np.linalg.norm(tool[3:]), atol=1e-9
            )
            with self.assertRaises(ValueError):
                method(frame="nonsense")

    def test_gripper_fraction_maps_to_the_finger_width(self) -> None:
        env = self.env
        self.assertAlmostEqual(
            env.gripper_width_from_fraction(0.0), env.CLOSE_GRIPPER_WIDTH_M
        )
        self.assertAlmostEqual(
            env.gripper_width_from_fraction(1.0), env.OPEN_GRIPPER_WIDTH_M
        )
        # Out-of-range device readings are clamped rather than extrapolated.
        self.assertAlmostEqual(
            env.gripper_width_from_fraction(-5.0), env.CLOSE_GRIPPER_WIDTH_M
        )
        self.assertAlmostEqual(
            env.gripper_width_from_fraction(5.0), env.OPEN_GRIPPER_WIDTH_M
        )

    def test_gripper_command_is_held_between_calls(self) -> None:
        env = self.env
        env.reset(episode_index=0)
        target = env.tool_pos.copy()
        env.step(target, gripper_width=env.CLOSE_GRIPPER_WIDTH_M)
        for _ in range(300):
            env.step(target)  # no width given
        self.assertAlmostEqual(env.gripper_width_cmd, env.CLOSE_GRIPPER_WIDTH_M)

    def test_grip_force_reports_a_squeeze_on_the_cube(self) -> None:
        env = ConveyorTeleop(
            seed=0, episode_index=0, belt_speed_m_per_s=0.0, respawn_object=False
        )
        try:
            cube = env.object_pos.copy()
            goal = np.array([cube[0], cube[1], cube[2] - 0.034])
            target = env.tool_pos.copy()
            width = env.OPEN_GRIPPER_WIDTH_M
            for index in range(4000):
                delta = goal - target
                distance = float(np.linalg.norm(delta))
                step = 0.25 * env.timestep
                target = goal if distance <= step else target + delta * (step / distance)
                if index > 2500:
                    width = max(
                        env.CLOSE_GRIPPER_WIDTH_M, width - 0.3 * env.timestep
                    )
                env.step(target, gripper_width=width)
            self.assertGreater(env.grip_force(), 5.0)
            self.assertGreater(float(np.linalg.norm(env.contact_force())), 0.0)
        finally:
            env.close()

    def test_every_reset_draws_a_new_belt_speed(self) -> None:
        env = ConveyorTeleop(seed=3, episode_index=0, respawn_object=False)
        try:
            speeds = []
            for _ in range(4):
                speeds.append(env.conveyor_speed_m_per_s)
                env.reset()
            self.assertEqual(len(set(speeds)), len(speeds))
            for speed in speeds:
                self.assertGreaterEqual(speed, 0.01)
                self.assertLessEqual(speed, 0.30)
        finally:
            env.close()

    def test_judge_state_is_reset_with_the_episode(self) -> None:
        env = ConveyorTeleop(
            seed=0, episode_index=0, belt_speed_m_per_s=0.30, respawn_object=False
        )
        try:
            target = env.tool_pos.copy()
            for _ in range(6000):
                env.step(target, gripper_width=env.OPEN_GRIPPER_WIDTH_M)
            self.assertFalse(env.success())
            self.assertEqual(env.termination_reason, "object_missed")
            env.reset()
            self.assertEqual(env.termination_reason, "running")
            self.assertFalse(env.success())
        finally:
            env.close()

    def test_recorder_channels_and_metadata_are_json_friendly(self) -> None:
        env = self.env
        channels = env.recorder_task_channels()
        for key in (
            "object_pose",
            "object_twist_world",
            "conveyor_speed_m_per_s",
            "grip_force",
            "gripper_width",
            "success",
        ):
            self.assertIn(key, channels)
        self.assertEqual(np.asarray(channels["object_pose"]).shape, (7,))
        self.assertEqual(np.asarray(channels["object_twist_world"]).shape, (6,))

        metadata = env.episode_metadata()
        self.assertEqual(metadata["task"], "conveyor_pick_place")
        self.assertAlmostEqual(
            metadata["conveyor_speed_m_per_s"], env.conveyor_speed_m_per_s
        )
        import json

        json.dumps(metadata)

    def test_arm_visual_modes_do_not_change_the_physics(self) -> None:
        env = self.env
        env.reset(episode_index=0)
        contype = env.model.geom_contype.copy()
        conaffinity = env.model.geom_conaffinity.copy()
        for mode in ("hidden", "ghost", "full"):
            env.set_arm_visual(mode)
            np.testing.assert_array_equal(env.model.geom_contype, contype)
            np.testing.assert_array_equal(env.model.geom_conaffinity, conaffinity)
        with self.assertRaises(ValueError):
            env.set_arm_visual("nonsense")
        env.set_arm_visual("full")

    def test_scene_cameras_render(self) -> None:
        env = self.env
        self.assertIn("third_person_camera", env.camera_names())
        self.assertIn("ur5e/wsg50/wrist_camera", env.camera_names())
        for kwargs in (
            {"camera": "third_person_camera"},
            {"quality": "collision"},
            {},
        ):
            render = env.make_camera(width=64, height=48, **kwargs)
            image = render()
            self.assertEqual(image.shape, (48, 64, 3))
        with self.assertRaises(ValueError):
            env.make_camera(quality="nonsense")


class PredictedFeelTest(unittest.TestCase):
    def test_flipup_defaults_reproduce_their_documented_margin(self) -> None:
        feel = predicted_feel(
            tool_kp=16000.0,
            scale=4.0,
            force_gain=3000.0 / (16000.0 * 4.0),
            damping=30.0,
            control_freq=1000.0,
            force_tau=0.002,
        )
        self.assertAlmostEqual(feel["k_handle_n_per_m"], 3000.0)
        self.assertAlmostEqual(feel["passivity_limit_n_per_m"], 12000.0)
        self.assertAlmostEqual(feel["margin"], 4.0)

    def test_halving_the_rendered_stiffness_doubles_the_margin(self) -> None:
        low = predicted_feel(16000.0, 4.0, 0.0234375, 30.0, 1000.0, 0.002)
        self.assertAlmostEqual(low["margin"], 8.0)


class ConveyorDatasetTest(unittest.TestCase):
    def test_recorder_writes_and_validates_a_conveyor_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "conveyor.zarr"
            with ConveyorTeleop(
                seed=0,
                episode_index=0,
                belt_speed_m_per_s=0.12,
                settle_seconds=0.0,
                respawn_object=False,
            ) as env:
                recorder = PyriteEpisodeRecorder(
                    dataset_path,
                    include_rgb=False,
                    min_samples=2,
                    wrench_filter_seconds=0.0,
                    schema_name=CONVEYOR_SCHEMA_NAME,
                )
                recorder.start_episode(env.episode_metadata())
                for index in range(3):
                    target = env.tool_pos + np.array([0.0, -0.0002, 0.0])
                    env.step(target, n_substeps=50, gripper_width=env.OPEN_GRIPPER_WIDTH_M)
                    recorded = recorder.record_sample(
                        env,
                        timestamp_ms=index * 50.0,
                        target_pos=target,
                        target_rotvec=None,
                        device_state=None,
                        sent_force=np.zeros(3),
                        image_rgb=None,
                        image_capture_time_s=None,
                    )
                    self.assertTrue(recorded)
                name = recorder.commit(
                    success=env.success(),
                    termination_reason=env.termination_reason,
                    episode_attrs={
                        "conveyor_speed_m_per_s": env.conveyor_speed_m_per_s
                    },
                )
            self.assertEqual(name, "episode_0")

            report = validate_pyrite_dataset(dataset_path)
            self.assertEqual(report["schema_name"], CONVEYOR_SCHEMA_NAME)
            self.assertEqual(report["episodes"], 1)
            self.assertEqual(report["episode_lengths"], [3])
            # The conveyor channels replaced the FlipUp book channels.
            self.assertIn("object_pose", report["keys"])
            self.assertIn("conveyor_speed_m_per_s", report["keys"])
            self.assertIn("grip_force", report["keys"])
            self.assertNotIn("book_pose", report["keys"])
            self.assertNotIn("book_angle_deg", report["keys"])

    def test_recorder_rejects_an_unknown_schema_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                PyriteEpisodeRecorder(
                    Path(temp_dir) / "bad.zarr", schema_name="not_a_schema"
                )


class CollectorTest(unittest.TestCase):
    """The collection loop itself, exercised through its --dry-run operator."""

    def test_pos_map_is_a_signed_permutation(self) -> None:
        from collect_conveyor import build_pos_map

        matrix = build_pos_map("-x,-y,z")
        np.testing.assert_allclose(
            matrix, [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        # Its transpose maps sim forces back to device axes, so resistance
        # opposes the motion that caused it.
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3))
        for bad in ("x,y", "x,x,z", "x,y,w"):
            with self.assertRaises(ValueError):
                build_pos_map(bad)

    def test_force_renderer_filters_slews_and_clamps(self) -> None:
        from collect_conveyor import ForceRenderer

        renderer = ForceRenderer(
            gain=0.1, tau=0.002, rate_limit=120.0, clip=8.0, timestep=0.001
        )
        # A step input arrives gradually, never faster than the slew limit.
        previous = np.zeros(3)
        for _ in range(50):
            sent = renderer(np.array([1000.0, 0.0, 0.0]))
            self.assertLessEqual(
                float(np.linalg.norm(sent - previous)), 120.0 * 0.001 + 1e-9
            )
            previous = sent
        for _ in range(5000):
            sent = renderer(np.array([1000.0, 0.0, 0.0]))
        self.assertLessEqual(float(np.linalg.norm(sent)), 8.0 + 1e-9)

    def test_control_freq_must_be_a_multiple_of_the_dataset_rate(self) -> None:
        from collect_conveyor import parse_args

        with self.assertRaises(SystemExit):
            parse_args(["--control-freq", "1000", "--dataset-hz", "30"])
        args = parse_args(["--control-freq", "1000", "--dataset-hz", "20"])
        self.assertEqual(args.record_stride, 50)
        # --dry-run implies auto-keep and turns real-time pacing off.
        args = parse_args(["--dry-run"])
        self.assertTrue(args.auto_keep)
        self.assertFalse(args.realtime)

    def test_dry_run_collects_a_validatable_episode(self) -> None:
        from collect_conveyor import main

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "conveyor.zarr"
            status = main(
                [
                    "--dry-run",
                    "--episodes",
                    "1",
                    "--auto-finish",
                    "--no-rgb",
                    "--max-episode-seconds",
                    "12",
                    "--conveyor-speed",
                    "0.15",
                    "--dataset",
                    str(dataset_path),
                ]
            )
            self.assertEqual(status, 0)
            report = validate_pyrite_dataset(dataset_path)
            self.assertEqual(report["schema_name"], CONVEYOR_SCHEMA_NAME)
            self.assertEqual(report["episodes"], 1)
            self.assertIn("conveyor_speed_m_per_s", report["keys"])

            import zarr

            episode = zarr.open(str(dataset_path), mode="r")["data"]["episode_0"]
            self.assertTrue(bool(episode.attrs["success"]))
            self.assertEqual(episode.attrs["termination_reason"], "success")
            np.testing.assert_allclose(
                np.unique(np.asarray(episode["conveyor_speed_m_per_s"])), [0.15]
            )


if __name__ == "__main__":
    unittest.main()
