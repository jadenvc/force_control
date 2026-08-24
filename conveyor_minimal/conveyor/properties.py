from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol


class UniformRandom(Protocol):
    def uniform(self, low: float, high: float) -> float:
        """Return a sample in the inclusive range [low, high]."""


@dataclass(frozen=True)
class ValueRange:
    """Inclusive range used to sample one property."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("Property range bounds must be finite")
        if self.minimum > self.maximum:
            raise ValueError(
                f"Property range minimum {self.minimum} exceeds maximum "
                f"{self.maximum}"
            )

    def sample(self, rng: UniformRandom) -> float:
        if self.minimum == self.maximum:
            return float(self.minimum)
        return float(rng.uniform(self.minimum, self.maximum))


# Belt speed is the one randomized quantity that does not need a model rebuild,
# so it lives outside CubeProperties and is resampled on every reset. The range
# is the source task's collection distribution (config/task/conveyor_pick_place
# .yaml: conveyor_speed_range [0.01, 0.3]).
DEFAULT_BELT_SPEED_M_PER_S = 0.18
DEFAULT_BELT_SPEED_RANGE = ValueRange(0.01, 0.30)


@dataclass(frozen=True)
class CubeProperties:
    """Compile-time physical parameters for the conveyed cube.

    These change the compiled model (collision size, rendered scale, mass,
    inertia), so unlike the belt speed they cannot be resampled on reset.
    """

    # The source asset's cube: a 5 cm box whose parent body carries 0.1 kg.
    mass_kg: float = 0.1
    half_extent_m: float = 0.025
    sliding_friction: float = 1.0
    torsional_friction: float = 0.3
    rolling_friction: float = 0.1

    def __post_init__(self) -> None:
        values = {
            "mass_kg": self.mass_kg,
            "half_extent_m": self.half_extent_m,
            "sliding_friction": self.sliding_friction,
            "torsional_friction": self.torsional_friction,
            "rolling_friction": self.rolling_friction,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        for name in ("mass_kg", "half_extent_m"):
            if values[name] <= 0.0:
                raise ValueError(f"{name} must be positive, got {values[name]}")
        for name in (
            "sliding_friction",
            "torsional_friction",
            "rolling_friction",
        ):
            if values[name] < 0.0:
                raise ValueError(f"{name} cannot be negative, got {values[name]}")

    @property
    def friction(self) -> tuple[float, float, float]:
        """MuJoCo sliding, torsional, and rolling friction coefficients."""
        return (
            self.sliding_friction,
            self.torsional_friction,
            self.rolling_friction,
        )

    @property
    def radius_m(self) -> float:
        """Circumscribed in-plane radius, used for spawn clearance."""
        return float(self.half_extent_m * math.sqrt(2.0))

    def summary(self) -> str:
        return (
            f"mass={self.mass_kg:.3f} kg, "
            f"friction={self.sliding_friction:.3f}/"
            f"{self.torsional_friction:.3f}/"
            f"{self.rolling_friction:.3f}, "
            f"edge={self.half_extent_m * 200.0:.2f} cm"
        )


DEFAULT_CUBE_PROPERTIES = CubeProperties()


@dataclass(frozen=True)
class CubePropertyRanges:
    """Uniform domain-randomization ranges for the cube."""

    mass_kg: ValueRange = ValueRange(0.05, 0.30)
    half_extent_m: ValueRange = ValueRange(0.020, 0.030)
    sliding_friction: ValueRange = ValueRange(0.6, 1.4)
    torsional_friction: ValueRange = ValueRange(0.1, 0.5)
    rolling_friction: ValueRange = ValueRange(0.02, 0.20)

    def sample(self, rng: UniformRandom) -> CubeProperties:
        return CubeProperties(
            mass_kg=self.mass_kg.sample(rng),
            half_extent_m=self.half_extent_m.sample(rng),
            sliding_friction=self.sliding_friction.sample(rng),
            torsional_friction=self.torsional_friction.sample(rng),
            rolling_friction=self.rolling_friction.sample(rng),
        )


DEFAULT_CUBE_PROPERTY_RANGES = CubePropertyRanges()


def sample_cube_properties(
    seed: int,
    ranges: CubePropertyRanges = DEFAULT_CUBE_PROPERTY_RANGES,
) -> CubeProperties:
    """Sample a reproducible cube configuration from ``seed``."""
    return ranges.sample(random.Random(seed))
