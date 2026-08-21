"""Launch rounded-cube lift teleoperation with the floating WSG50."""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

from floating_cube_lift_teleop import FloatingCubeLiftTeleop
from teleop_flipup import main


if __name__ == "__main__":
    main(env_class=FloatingCubeLiftTeleop)
