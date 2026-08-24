from __future__ import annotations

import math
import random

import pytest

from conveyor.properties import (
    DEFAULT_BELT_SPEED_RANGE,
    DEFAULT_CUBE_PROPERTIES,
    DEFAULT_CUBE_PROPERTY_RANGES,
    CubeProperties,
    ValueRange,
    sample_cube_properties,
)


def test_value_range_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        ValueRange(0.3, 0.1)


def test_value_range_degenerate_range_needs_no_rng():
    assert ValueRange(0.2, 0.2).sample(None) == 0.2  # type: ignore[arg-type]


def test_value_range_samples_inside_bounds():
    rng = random.Random(0)
    for _ in range(100):
        value = DEFAULT_BELT_SPEED_RANGE.sample(rng)
        assert DEFAULT_BELT_SPEED_RANGE.minimum <= value
        assert value <= DEFAULT_BELT_SPEED_RANGE.maximum


def test_default_cube_matches_source_asset():
    assert DEFAULT_CUBE_PROPERTIES.mass_kg == pytest.approx(0.1)
    assert DEFAULT_CUBE_PROPERTIES.half_extent_m == pytest.approx(0.025)
    assert DEFAULT_CUBE_PROPERTIES.friction == (1.0, 0.3, 0.1)


def test_default_belt_speed_range_matches_source_collection():
    assert DEFAULT_BELT_SPEED_RANGE.minimum == pytest.approx(0.01)
    assert DEFAULT_BELT_SPEED_RANGE.maximum == pytest.approx(0.30)


def test_cube_radius_circumscribes_the_box():
    properties = CubeProperties(half_extent_m=0.03)
    assert properties.radius_m == pytest.approx(0.03 * math.sqrt(2.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass_kg": 0.0},
        {"mass_kg": -1.0},
        {"half_extent_m": 0.0},
        {"sliding_friction": -0.1},
        {"mass_kg": float("nan")},
    ],
)
def test_cube_properties_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CubeProperties(**kwargs)


def test_cube_sampling_is_reproducible_and_in_range():
    first = sample_cube_properties(7)
    assert first == sample_cube_properties(7)
    assert first != sample_cube_properties(8)

    ranges = DEFAULT_CUBE_PROPERTY_RANGES
    for name in (
        "mass_kg",
        "half_extent_m",
        "sliding_friction",
        "torsional_friction",
        "rolling_friction",
    ):
        value = getattr(first, name)
        bounds = getattr(ranges, name)
        assert bounds.minimum <= value <= bounds.maximum
