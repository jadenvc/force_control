from __future__ import annotations

import random
import unittest

from flipup.physical_properties import (
    DEFAULT_PHYSICAL_PROPERTIES,
    DEFAULT_PHYSICAL_PROPERTY_RANGES,
    PhysicalProperties,
    ValueRange,
    sample_physical_properties,
)


class PhysicalPropertiesTest(unittest.TestCase):
    def test_seeded_sampling_is_reproducible_and_in_range(self) -> None:
        first = sample_physical_properties(17)
        second = sample_physical_properties(17)

        self.assertEqual(first, second)
        for field_name in first.__dataclass_fields__:
            value = getattr(first, field_name)
            value_range = getattr(DEFAULT_PHYSICAL_PROPERTY_RANGES, field_name)
            self.assertGreaterEqual(value, value_range.minimum)
            self.assertLessEqual(value, value_range.maximum)

    def test_ranges_accept_any_uniform_random_implementation(self) -> None:
        properties = DEFAULT_PHYSICAL_PROPERTY_RANGES.sample(random.Random(3))
        self.assertIsInstance(properties, PhysicalProperties)

    def test_default_summary_exposes_units_and_all_dimensions(self) -> None:
        summary = DEFAULT_PHYSICAL_PROPERTIES.summary()
        self.assertIn("1.375 kg", summary)
        self.assertIn("15.00 x 10.00 x 2.50 cm", summary)

    def test_invalid_properties_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "mass_kg must be positive"):
            PhysicalProperties(mass_kg=0.0)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            PhysicalProperties(sliding_friction=-0.1)
        with self.assertRaisesRegex(ValueError, "greater than width_m"):
            PhysicalProperties(length_m=0.09, width_m=0.10)

    def test_invalid_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds maximum"):
            ValueRange(2.0, 1.0)


if __name__ == "__main__":
    unittest.main()
