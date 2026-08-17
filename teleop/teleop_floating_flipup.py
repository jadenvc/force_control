"""Run flip-up teleoperation with only a dynamical floating WSG50 gripper."""

import os

# Must precede dm_control imports in floating_flipup_teleop.
os.environ.setdefault("MUJOCO_GL", "osmesa")

from floating_flipup_teleop import FloatingFlipUpTeleop
from teleop_flipup import main


if __name__ == "__main__":
    main(env_class=FloatingFlipUpTeleop)
