import unittest
from myhdl import *
import sys
import os

# Import your module and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.hdl.components.fp8_e4m3_mult import fp8_e4m3_multiply
from src.utils.fp_defs import E4M3Format
from tests.utils.hdl_test_utils import test_runner
from tests.utils.fp8_helpers import float_to_fp8, fp8_to_float


class TestFP8E4M3Multiply(unittest.TestCase):
    """Test case for the 8-bit E4M3 floating-point multiplier."""

    def setUp(self):
        """Setup common signals and parameters for all tests."""
        # Common signals
        self.clk = Signal(bool(0))
        self.rst = ResetSignal(0, active=1, isasync=False)
        self.input_a = Signal(intbv(0)[E4M3Format.WIDTH :])
        self.input_b = Signal(intbv(0)[E4M3Format.WIDTH :])
        self.output_z = Signal(intbv(0)[E4M3Format.WIDTH :])
        self.start = Signal(bool(0))
        self.done = Signal(bool(0))
        self.sim = None

    def tearDown(self):
        """Clean up after each test."""
        if self.sim is not None:
            self.sim.quit()

    def create_fp8_multiplier(self):
        """Helper to create multiplier instance with current signals."""
        # Updated to use the new interface with start and done signals
        return fp8_e4m3_multiply(
            self.input_a,
            self.input_b,
            self.output_z,
            self.start,
            self.done,
            self.clk,
            self.rst,
        )

    def run_multiplication_test(
        self, a_val, b_val, expected, test_name, compare_bits=True
    ):
        """
        Helper method to run a single multiplication test.

        Args:
            a_val: First operand (float or raw bits)
            b_val: Second operand (float or raw bits)
            expected: Expected result (float or raw bits)
            test_name: Name or description of the test
            compare_bits: If True, compare raw bit patterns, otherwise check float values
        """
        # Convert float values to E4M3 format
        a_fp8 = a_val if isinstance(a_val, int) else float_to_fp8(a_val)
        b_fp8 = b_val if isinstance(b_val, int) else float_to_fp8(b_val)
        expected_fp8 = expected if isinstance(expected, int) else float_to_fp8(expected)

        # Print test information
        a_float = fp8_to_float(a_fp8) if isinstance(a_val, int) else a_val
        b_float = fp8_to_float(b_fp8) if isinstance(b_val, int) else b_val
        expected_float = (
            fp8_to_float(expected_fp8) if isinstance(expected, int) else expected
        )

        print(f"\n{test_name}: {a_float} * {b_float} = {expected_float}")
        print(f"Input A: {a_float} => 0x{a_fp8:02x}")
        print(f"Input B: {b_float} => 0x{b_fp8:02x}")
        print(f"Expected: {expected_float} => 0x{expected_fp8:02x}")

        # Set inputs and start computation
        self.input_a.next = a_fp8
        self.input_b.next = b_fp8
        self.start.next = 1
        yield self.clk.posedge
        self.start.next = 0

        # Wait for completion
        while not self.done:
            yield self.clk.posedge

        # Get result
        result_fp8 = int(self.output_z)
        result_float = fp8_to_float(result_fp8)
        print(f"Result: 0x{result_fp8:02x} => {result_float}")

        # Verify result based on comparison mode
        if compare_bits:
            assert (
                result_fp8 == expected_fp8
            ), f"Expected 0x{expected_fp8:02x}, got 0x{result_fp8:02x} for {a_float} * {b_float}"
        else:
            assert (
                abs(result_float - expected_float) < 0.1
            ), f"Expected {expected_float}, got {result_float} for {a_float} * {b_float}"

        # Wait an extra cycle between tests
        yield self.clk.posedge

    def testBasicMultiplication(self):
        """Test basic multiplication with two simple values."""

        @instance
        def test_sequence():
            # Reset the system
            self.rst.next = 1
            yield self.clk.posedge
            yield self.clk.posedge
            self.rst.next = 0
            yield self.clk.posedge

            # Test case 1: Basic positive multiplication
            yield from self.run_multiplication_test(1.5, 2.0, 3.0, "Test 1")

            # Test case 2: Multiplication with negative number
            yield from self.run_multiplication_test(2.0, -1.5, -3.0, "Test 2")

            # Test case 3: Multiplication requiring exponent adjustment
            yield from self.run_multiplication_test(4.0, 0.25, 1.0, "Test 3")

            # Test case 4: Zero handling tests
            yield from self.run_multiplication_test(0.0, 2.0, 0.0, "Test 4.1")
            yield from self.run_multiplication_test(3.0, 0.0, 0.0, "Test 4.2")
            yield from self.run_multiplication_test(0.0, 0.0, 0.0, "Test 4.3")

        # Run simulation
        self.sim = test_runner(
            self.create_fp8_multiplier,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="fp8_e4m3_multiply",
            vcd_output=True,
            duration=1000,
        )

    def testEdgeCases(self):
        """Test edge cases for the E4M3 multiplier."""

        @instance
        def test_sequence():
            # Reset the system
            self.rst.next = 1
            yield self.clk.posedge
            yield self.clk.posedge
            self.rst.next = 0
            yield self.clk.posedge

            # Test case 1: Min positive value * Min positive value
            # 2^-6 * 2^-6 = 2^-12 (would underflow)
            min_positive = 0x01  # Smallest representable positive number
            yield from self.run_multiplication_test(
                min_positive, min_positive, 0x00, "Min positive * Min positive"
            )

            # Test case 2: Max finite value * Min positive value
            # This should still be representable
            max_finite = 0x7E  # Largest normal value (01111110) - just below NaN
            yield from self.run_multiplication_test(
                max_finite,
                min_positive,
                0x01,
                "Max finite * Min positive",
                compare_bits=False,
            )

            # Test case 3: Max finite value * 2.0
            # Should saturate to max finite value
            yield from self.run_multiplication_test(
                max_finite, 0x40, max_finite, "Max finite * 2.0"
            )

            # Test case 4: Min negative value * Min negative value
            # Should give positive result (negative * negative = positive)
            min_negative = 0xFE  # Largest negative number (11111110)
            yield from self.run_multiplication_test(
                min_negative, min_negative, max_finite, "Min negative * Min negative"
            )

            # Test case 5: Min negative value * Max finite value
            # Should give min negative value (saturate)
            yield from self.run_multiplication_test(
                min_negative, max_finite, min_negative, "Min negative * Max finite"
            )

            # Test case 6: Numbers with large exponent difference
            # 64.0 * 0.0625 = 4.0
            large_number = 0x70  # 64.0
            small_number = 0x10  # 0.0625
            yield from self.run_multiplication_test(
                large_number, small_number, 0x48, "Large exponent difference"
            )

            # Test case 7: Subnormal numbers (denormalized)
            # 2^-6 * 0.5 = 2^-7 (which is subnormal in E4M3)
            yield from self.run_multiplication_test(
                0x01, 0x30, 0x00, "Subnormal result", compare_bits=True
            )

            # Test case 8: Exact boundary value where result needs to be normalized
            # 0.5 * 0.5 = 0.25 (requires normalization)
            yield from self.run_multiplication_test(
                0x30, 0x30, 0x20, "Normalization boundary"
            )

            # Test case 9: Rounding case
            # A case where rounding should occur
            # 3.0 * 1.5 = 4.5, which should round correctly
            yield from self.run_multiplication_test(0x44, 0x3C, 0x48, "Rounding case")

            # Test case 10: NaN handling
            # Any operation with NaN should result in NaN
            nan_value = 0x7F  # NaN in E4M3 (01111111)
            # NaN * normal number = NaN
            yield from self.run_multiplication_test(
                nan_value, 0x40, nan_value, "NaN * normal number"
            )
            # NaN * NaN = NaN
            yield from self.run_multiplication_test(
                nan_value, nan_value, nan_value, "NaN * NaN"
            )

            # Test case 11: Multiplication that causes exponent overflow
            # Result should saturate to max value (not NaN)
            large_exp_a = 0x70  # 64.0
            large_exp_b = 0x70  # 64.0
            yield from self.run_multiplication_test(
                large_exp_a, large_exp_b, max_finite, "Exponent overflow"
            )

            # Test case 12: Multiplication that causes exponent underflow
            # Result should normalize to smallest representable value or zero
            small_mul_a = 0x10  # 0.0625
            small_mul_b = 0x10  # 0.0625
            yield from self.run_multiplication_test(
                small_mul_a, small_mul_b, 0x00, "Exponent underflow"
            )

            # Test case 13: Mixed sign multiplication
            # Testing a positive * negative = negative
            yield from self.run_multiplication_test(
                0x40, 0xC0, 0xC0, "Mixed sign multiplication"
            )

            # Test case 14: Mixed sign near zero
            # Testing a case where tiny values of opposite sign multiply
            yield from self.run_multiplication_test(
                0x01, 0x81, 0x81, "Mixed sign tiny values", compare_bits=False
            )

        # Run simulation
        self.sim = test_runner(
            self.create_fp8_multiplier,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="fp8_e4m3_multiply_edge_cases",
            vcd_output=True,
            duration=1000,
        )

    def testAccuracy(self):
        """Test multiplication accuracy for a range of values."""

        @instance
        def test_sequence():
            # Reset the system
            self.rst.next = 1
            yield self.clk.posedge
            yield self.clk.posedge
            self.rst.next = 0
            yield self.clk.posedge

            # Test a range of values with known results
            test_cases = [
                # (a, b, expected, name)
                (1.0, 1.0, 1.0, "Unity * Unity"),
                (2.0, 0.5, 1.0, "Reciprocal multiplication"),
                (4.0, 4.0, 16.0, "Power of 2 multiplication"),
                (3.0, 3.0, 9.0, "Non-power of 2 multiplication"),
                (1.5, 1.5, 2.25, "Fractional multiplication"),
                (-2.0, 3.0, -6.0, "Negative * Positive"),
                (-3.0, -2.0, 6.0, "Negative * Negative"),
                (16.0, 0.125, 2.0, "Large * Small"),
                (0.25, 0.5, 0.125, "Small * Small"),
            ]

            for a, b, expected, name in test_cases:
                yield from self.run_multiplication_test(
                    a, b, expected, name, compare_bits=False
                )

        # Run simulation
        self.sim = test_runner(
            self.create_fp8_multiplier,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="fp8_e4m3_multiply_accuracy",
            vcd_output=True,
            duration=2000,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
