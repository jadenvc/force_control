from __future__ import annotations

import argparse
import random
from dataclasses import replace

from .controls import show_physics_controls
from .heuristic import run_flipup
from .physical_properties import (
    DEFAULT_PHYSICAL_PROPERTIES,
    DEFAULT_PHYSICAL_PROPERTY_RANGES,
    PhysicalProperties,
)


def _physical_properties_from_args(
    args: argparse.Namespace,
    rng: random.Random,
) -> PhysicalProperties:
    if args.randomize_physics:
        properties = DEFAULT_PHYSICAL_PROPERTY_RANGES.sample(rng)
    else:
        properties = DEFAULT_PHYSICAL_PROPERTIES

    overrides = {
        "mass_kg": args.book_mass,
        "sliding_friction": args.book_friction,
        "torsional_friction": args.book_torsional_friction,
        "rolling_friction": args.book_rolling_friction,
        "length_m": args.book_length,
        "width_m": args.book_width,
        "thickness_m": args.book_thickness,
    }
    return replace(
        properties,
        **{name: value for name, value in overrides.items() if value is not None},
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the standalone scripted MuJoCo FlipUp task."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the interactive MuJoCo viewer (default: enabled).",
    )
    parser.add_argument(
        "--success-threshold-deg",
        type=float,
        default=15.0,
    )
    physics_group = parser.add_argument_group("book physical properties")
    physics_group.add_argument(
        "--physics-controls",
        action="store_true",
        help="Open draggable physical-property sliders before the simulation.",
    )
    physics_group.add_argument(
        "--randomize-physics",
        action="store_true",
        help=(
            "Uniformly sample mass, friction, and dimensions from the default "
            "ranges using --seed."
        ),
    )
    physics_group.add_argument("--book-mass", type=float, metavar="KG")
    physics_group.add_argument(
        "--book-friction",
        type=float,
        metavar="COEFFICIENT",
        help="Override the book's sliding friction.",
    )
    physics_group.add_argument(
        "--book-torsional-friction",
        type=float,
        metavar="COEFFICIENT",
    )
    physics_group.add_argument(
        "--book-rolling-friction",
        type=float,
        metavar="COEFFICIENT",
    )
    physics_group.add_argument("--book-length", type=float, metavar="METERS")
    physics_group.add_argument("--book-width", type=float, metavar="METERS")
    physics_group.add_argument("--book-thickness", type=float, metavar="METERS")
    args = parser.parse_args()

    physics_rng = random.Random(args.seed)
    try:
        physical_properties = _physical_properties_from_args(args, physics_rng)
    except ValueError as exc:
        parser.error(str(exc))

    if args.physics_controls:
        try:
            selected_properties = show_physics_controls(
                physical_properties,
                DEFAULT_PHYSICAL_PROPERTY_RANGES,
                rng=physics_rng,
            )
        except RuntimeError as exc:
            parser.error(str(exc))
        if selected_properties is None:
            print("FlipUp run cancelled.")
            return 0
        physical_properties = selected_properties

    result = run_flipup(
        seed=args.seed,
        show_viewer=args.viewer,
        success_threshold_deg=args.success_threshold_deg,
        physical_properties=physical_properties,
    )
    return 0 if result.success else 1
