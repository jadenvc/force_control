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
    """Inclusive range used to sample one physical property."""

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


@dataclass(frozen=True)
class PhysicalProperties:
    """Physical parameters for the movable book."""

    # The legacy asset's 1 kg collision box plus its inferred 0.375 kg visual
    # mesh produced a 1.375 kg compiled body. Keeping that total here preserves
    # the original task while making future mass selections exact.
    mass_kg: float = 1.375
    sliding_friction: float = 0.12
    torsional_friction: float = 0.002
    rolling_friction: float = 0.0001
    length_m: float = 0.15
    width_m: float = 0.10
    thickness_m: float = 0.025

    def __post_init__(self) -> None:
        values = {
            "mass_kg": self.mass_kg,
            "sliding_friction": self.sliding_friction,
            "torsional_friction": self.torsional_friction,
            "rolling_friction": self.rolling_friction,
            "length_m": self.length_m,
            "width_m": self.width_m,
            "thickness_m": self.thickness_m,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")

        if self.mass_kg <= 0.0:
            raise ValueError(f"mass_kg must be positive, got {self.mass_kg}")
        for name in (
            "sliding_friction",
            "torsional_friction",
            "rolling_friction",
        ):
            value = values[name]
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative, got {value}")
        for name in ("length_m", "width_m", "thickness_m"):
            value = values[name]
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.length_m <= self.width_m:
            raise ValueError(
                "length_m must be greater than width_m because the FlipUp "
                "trajectory treats the local x-axis as the book's long axis"
            )

    @property
    def friction(self) -> tuple[float, float, float]:
        """MuJoCo sliding, torsional, and rolling friction coefficients."""
        return (
            self.sliding_friction,
            self.torsional_friction,
            self.rolling_friction,
        )

    def summary(self) -> str:
        return (
            f"mass={self.mass_kg:.3f} kg, "
            f"friction={self.sliding_friction:.4f}/"
            f"{self.torsional_friction:.5f}/"
            f"{self.rolling_friction:.6f}, "
            f"size={self.length_m * 100.0:.2f} x "
            f"{self.width_m * 100.0:.2f} x "
            f"{self.thickness_m * 100.0:.2f} cm"
        )


DEFAULT_PHYSICAL_PROPERTIES = PhysicalProperties()


@dataclass(frozen=True)
class PhysicalPropertyRanges:
    """Uniform domain-randomization ranges for the book."""

    mass_kg: ValueRange = ValueRange(0.4, 2.0)
    sliding_friction: ValueRange = ValueRange(0.05, 0.60)
    torsional_friction: ValueRange = ValueRange(0.0005, 0.0100)
    rolling_friction: ValueRange = ValueRange(0.00002, 0.00100)
    length_m: ValueRange = ValueRange(0.12, 0.20)
    width_m: ValueRange = ValueRange(0.07, 0.11)
    thickness_m: ValueRange = ValueRange(0.015, 0.040)

    def sample(self, rng: UniformRandom) -> PhysicalProperties:
        return PhysicalProperties(
            mass_kg=self.mass_kg.sample(rng),
            sliding_friction=self.sliding_friction.sample(rng),
            torsional_friction=self.torsional_friction.sample(rng),
            rolling_friction=self.rolling_friction.sample(rng),
            length_m=self.length_m.sample(rng),
            width_m=self.width_m.sample(rng),
            thickness_m=self.thickness_m.sample(rng),
        )


DEFAULT_PHYSICAL_PROPERTY_RANGES = PhysicalPropertyRanges()


def sample_physical_properties(
    seed: int,
    ranges: PhysicalPropertyRanges = DEFAULT_PHYSICAL_PROPERTY_RANGES,
) -> PhysicalProperties:
    """Sample a reproducible physical configuration from ``seed``."""
    return ranges.sample(random.Random(seed))
