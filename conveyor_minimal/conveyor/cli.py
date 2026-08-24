from __future__ import annotations

import argparse
from dataclasses import replace

from .heuristic import run_conveyor
from .properties import (
    DEFAULT_BELT_SPEED_RANGE,
    DEFAULT_CUBE_PROPERTIES,
    DEFAULT_CUBE_PROPERTY_RANGES,
    CubeProperties,
    ValueRange,
    sample_cube_properties,
)


def _cube_properties_from_args(args: argparse.Namespace) -> CubeProperties | None:
    overrides = {
        "mass_kg": args.cube_mass,
        "half_extent_m": None if args.cube_edge is None else args.cube_edge / 2.0,
        "sliding_friction": args.cube_friction,
    }
    overrides = {name: value for name, value in overrides.items() if value is not None}
    if not overrides:
        return None

    base = (
        sample_cube_properties(args.seed, DEFAULT_CUBE_PROPERTY_RANGES)
        if args.randomize_cube
        else DEFAULT_CUBE_PROPERTIES
    )
    return replace(base, **overrides)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the standalone scripted MuJoCo conveyor pick-place task."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Which episode of --seed to run. Selects the belt speed, layout "
        "jitter and cube spawn pose (default: 0).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Run this many consecutive episode indices and report the tally.",
    )
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the interactive MuJoCo viewer (default: enabled).",
    )
    parser.add_argument("--time-limit", type=float, default=30.0, metavar="SECONDS")

    belt_group = parser.add_argument_group("conveyor belt")
    belt_group.add_argument(
        "--conveyor-speed",
        type=float,
        metavar="M_PER_S",
        help="Pin the belt speed instead of resampling it on every reset.",
    )
    belt_group.add_argument(
        "--conveyor-speed-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(DEFAULT_BELT_SPEED_RANGE.minimum, DEFAULT_BELT_SPEED_RANGE.maximum),
        help="Range the per-reset belt speed is drawn from "
        f"(default: {DEFAULT_BELT_SPEED_RANGE.minimum} "
        f"{DEFAULT_BELT_SPEED_RANGE.maximum}).",
    )
    belt_group.add_argument(
        "--randomize-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Jitter the belt and bin position on every reset (default: enabled).",
    )

    cube_group = parser.add_argument_group("cube physical properties")
    cube_group.add_argument(
        "--randomize-cube",
        action="store_true",
        help="Sample cube mass, size and friction from the default ranges using "
        "--seed. Uses a random stream separate from the episode randomization.",
    )
    cube_group.add_argument("--cube-mass", type=float, metavar="KG")
    cube_group.add_argument("--cube-edge", type=float, metavar="METERS")
    cube_group.add_argument(
        "--cube-friction",
        type=float,
        metavar="COEFFICIENT",
        help="Override the cube's sliding friction, which is also the grip the "
        "belt has on it.",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    try:
        speed_range = ValueRange(*args.conveyor_speed_range)
        cube_properties = _cube_properties_from_args(args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    successes = 0
    for offset in range(args.episodes):
        result = run_conveyor(
            seed=args.seed,
            episode_index=args.episode_index + offset,
            show_viewer=args.viewer,
            belt_speed_m_per_s=args.conveyor_speed,
            belt_speed_range=speed_range,
            randomize_layout=args.randomize_layout,
            cube_properties=cube_properties,
            randomize_cube=args.randomize_cube and cube_properties is None,
            time_limit_s=args.time_limit,
        )
        successes += int(result.success)
        if result.viewer_closed:
            break

    if args.episodes > 1:
        print(f"{successes}/{args.episodes} episodes succeeded")
    return 0 if successes == args.episodes else 1
