import unittest
import math
import sys
import os

# Import your module and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from tests.utils.fp8_helpers import E4M3Format
from tests.utils.fp8_helpers import float_to_fp8, fp8_to_float


class TestE4M3FormatRange(unittest.TestCase):
    """
    Unit tests specifically for the value range of E4M3Format.
    """

    def test_float_value_range(self):
        """Test the full range of representable float values."""
        # Maximum positive value
        max_pos = E4M3Format(0x7E)
        self.assertAlmostEqual(max_pos.to_float(), 448.0, delta=1.0)

        # Create from float
        self.assertEqual(E4M3Format(448.0).raw_value, 0x7E)

        # Maximum negative value
        max_neg = E4M3Format(0xFE)
        self.assertAlmostEqual(max_neg.to_float(), -448.0, delta=1.0)

        # Create from float
        self.assertEqual(E4M3Format(-448.0).raw_value, 0xFE)

    def test_value_clamping(self):
        """Test that values outside the representable range are clamped."""
        # Values above max are clamped to max
        self.assertEqual(E4M3Format(1000.0).raw_value, 0x7E)
        self.assertEqual(E4M3Format(float("inf")).raw_value, 0x7E)

        # Values below min are clamped to min
        self.assertEqual(E4M3Format(-1000.0).raw_value, 0xFE)
        self.assertEqual(E4M3Format(float("-inf")).raw_value, 0xFE)

    def test_small_value_representation(self):
        """Test representation of small values near zero."""
        # Smallest positive normal value (2^-6)
        small_pos = E4M3Format(0x08)
        self.assertAlmostEqual(small_pos.to_float(), 2**-6, delta=0.0001)

        # Smallest negative normal value (-2^-6)
        small_neg = E4M3Format(0x88)
        self.assertAlmostEqual(small_neg.to_float(), -(2**-6), delta=0.0001)

        # Smallest positive denormal (2^-6 * 1/8)
        min_denormal = E4M3Format(0x01)
        self.assertAlmostEqual(min_denormal.to_float(), 2**-6 * (1 / 8), delta=0.0001)

    def test_conversion_accuracy(self):
        """Test that values convert to their expected E4M3 representations."""
        # Define (input value, expected output value) pairs
        # These represent the closest representable value in E4M3 format
        test_cases = [
            # Input           Expected
            (0.0, 0.0),
            (1.0, 1.0),
            (-1.0, -1.0),
            (2.0, 2.0),
            (-2.0, -2.0),
            (4.0, 4.0),
            (-4.0, -4.0),
            (8.0, 8.0),
            (-8.0, -8.0),
            (16.0, 16.0),
            (-16.0, -16.0),
            (32.0, 32.0),
            (-32.0, -32.0),
            (64.0, 64.0),
            (-64.0, -64.0),
            (128.0, 128.0),
            (-128.0, -128.0),
            (256.0, 256.0),  # Representable in E4M3
            (-256.0, -256.0),
            (384.0, 448.0),  # Rounds to max value in E4M3
            (448.0, 448.0),  # Max representable value in E4M3
            (-448.0, -448.0),  # Min representable value in E4M3
            (0.5, 0.5),
            (-0.5, -0.5),
            (0.25, 0.25),
            (-0.25, -0.25),
            (0.125, 0.125),
            (-0.125, -0.125),
            (0.0625, 0.0625),  # Smallest normal value
            (-0.0625, -0.0625),
            (0.03125, 0.03125),  # Representable denormal values
            (-0.03125, -0.03125),
            (0.015625, 0.015625),
            (-0.015625, -0.015625),
            (0.001, 0.001953),  # Smallest possible
            (-0.001, -0.001953),
            (500.0, 448.0),  # Clipped to max
            (-500.0, -448.0),  # Clipped to min
            (1.75, 1.75),  # Exactly representable
            (1.8, 1.75),  # Rounds to closest representable
            (1.9, 1.875),  # Rounds to closest representable
        ]

        for input_val, expected_val in test_cases:
            e4m3 = E4M3Format(input_val)
            round_trip = e4m3.to_float()

            # For expected zero, check exact equality
            if expected_val == 0:
                self.assertEqual(
                    round_trip,
                    expected_val,
                    f"Failed for {input_val}: expected {expected_val}, got {round_trip}",
                )
            # For NaN, check isnan
            elif isinstance(expected_val, str) and expected_val.lower() == "nan":
                self.assertTrue(
                    math.isnan(round_trip),
                    f"Failed for {input_val}: expected NaN, got {round_trip}",
                )
            # For normal values, use assertAlmostEqual with small delta
            else:
                self.assertAlmostEqual(
                    round_trip,
                    expected_val,
                    delta=1e-6,
                    msg=f"Failed for {input_val}: expected {expected_val}, got {round_trip}",
                )

    def test_exact_bit_patterns(self):
        """Test specific bit patterns and their float values."""
        test_cases = [
            # (hex value, expected float)
            (0x00, 0.0),  # Zero
            (0x80, -0.0),  # Negative zero
            (0x38, 1.0),  # One
            (0xB8, -1.0),  # Negative one
            (0x40, 2.0),  # Two
            (0xC0, -2.0),  # Negative two
            (0x7E, 448.0),  # Max positive
            (0xFE, -448.0),  # Max negative
            (0x7F, float("nan")),  # NaN
        ]

        for hex_val, expected_float in test_cases:
            e4m3 = E4M3Format(hex_val)
            if math.isnan(expected_float):
                self.assertTrue(math.isnan(e4m3.to_float()))
            else:
                self.assertAlmostEqual(
                    e4m3.to_float(),
                    expected_float,
                    delta=0.1,
                    msg=f"Failed for 0x{hex_val:02x}: expected {expected_float}, got {e4m3.to_float()}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
