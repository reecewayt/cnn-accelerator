import unittest
from myhdl import *
import sys
import os

# Import your module and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.hdl.components.fp8_e4m3_add import fp8_e4m3_add
from src.utils.fp_defs import E4M3Format
from tests.utils.hdl_test_utils import test_runner
from tests.utils.fp8_helpers import float_to_fp8, fp8_to_float


class TestFP8E4M3Add(unittest.TestCase):
    """Test case for the 8-bit E4M3 floating-point adder."""

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

    def create_fp8_adder(self):
        """Helper to create adder instance with current signals."""
        return fp8_e4m3_add(
            self.input_a,
            self.input_b,
            self.output_z,
            self.start,
            self.done,
            self.clk,
            self.rst,
        )

    def run_addition_test(self, a_val, b_val, expected, test_name, compare_bits=True):
        """
        Helper method to run a single addition test.

        Args:
            a_val: First operand (float)
            b_val: Second operand (float)
            expected: Expected result (float)
            test_name: Name or description of the test
            compare_bits: If True, compare raw bit patterns, otherwise check float values
        """
        # Convert float values to E4M3 format
        a_fp8 = float_to_fp8(a_val)
        b_fp8 = float_to_fp8(b_val)
        expected_fp8 = float_to_fp8(expected)

        # Print test information
        print(f"\n{test_name}: {a_val} + {b_val} = {expected}")
        print(f"Input A: {a_val} => 0x{a_fp8:02x}")
        print(f"Input B: {b_val} => 0x{b_fp8:02x}")
        print(f"Expected: {expected} => 0x{expected_fp8:02x}")

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
            ), f"Expected 0x{expected_fp8:02x}, got 0x{result_fp8:02x} for {a_val} + {b_val}"
        else:
            assert (
                abs(result_float - expected) < 0.1
            ), f"Expected {expected}, got {result_float} for {a_val} + {b_val}"

        # Wait an extra cycle between tests
        yield self.clk.posedge

    def testBasicAddition(self):
        """Test basic addition with two simple values."""

        @instance
        def test_sequence():
            # Reset the system
            self.rst.next = 1
            yield self.clk.posedge
            yield self.clk.posedge
            self.rst.next = 0
            yield self.clk.posedge

            # Test case 1: Basic positive addition
            yield from self.run_addition_test(1.5, 2.0, 3.5, "Test 1")

            # Test case 2: Addition with negative number
            yield from self.run_addition_test(2.0, -1.5, 0.5, "Test 2")

            # Test case 3: Addition requiring alignment
            yield from self.run_addition_test(4.0, 0.25, 4.25, "Test 3")

            # Test case 4: Zero handling tests
            yield from self.run_addition_test(0.0, 2.0, 2.0, "Test 4.1")
            yield from self.run_addition_test(3.0, 0.0, 3.0, "Test 4.2")
            yield from self.run_addition_test(0.0, 0.0, 0.0, "Test 4.3")

        # Run simulation
        self.sim = test_runner(
            self.create_fp8_adder,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="fp8_e4m3_add",
            vcd_output=True,
            duration=1000,
        )

    def testEdgeCases(self):
        """Test edge cases for the E4M3 adder."""

        @instance
        def test_sequence():
            # Reset the system
            self.rst.next = 1
            yield self.clk.posedge
            yield self.clk.posedge
            self.rst.next = 0
            yield self.clk.posedge

            # Test case 1: Min positive value + Min positive value
            # 2^-6 + 2^-6 = 2^-5
            min_positive = 0x01  # Smallest representable positive number
            yield from self.run_addition_test(
                min_positive, min_positive, 0x02, "Min positive + Min positive"
            )

            # Test case 2: Max finite value + Min positive value
            # Should remain max finite value
            max_finite = 0x7E  # Largest normal value (01111110) - just below NaN
            # TODO: Next time check the best way to handle this. Current converts to NaN. should it
            # be 0x7E? No, it should be the max value.
            yield from self.run_addition_test(
                max_finite, min_positive, max_finite, "Max finite + Min positive"
            )

            # Test case 3: Max finite value + Max finite value
            # Should saturate to max finite value
            yield from self.run_addition_test(
                max_finite, max_finite, max_finite, "Max finite + Max finite"
            )

            # Test case 4: Min negative value + Min negative value
            # Should saturate to min negative value
            min_negative = 0xFE  # Largest negative number (11111110)
            yield from self.run_addition_test(
                min_negative, min_negative, min_negative, "Min negative + Min negative"
            )

            # Test case 5: Min negative value + Max finite value
            # These should cancel out approximately to zero
            yield from self.run_addition_test(
                min_negative, max_finite, 0x00, "Min negative + Max finite"
            )

            # Test case 6: Numbers with large exponent difference
            # 64 + 0.0625 = 64 (due to quantization)
            large_number = 0x70  # 64.0
            small_number = 0x10  # 0.0625
            yield from self.run_addition_test(
                large_number, small_number, large_number, "Large exponent difference"
            )

            # Test case 7: Subnormal numbers (denormalized)
            # 2^-6 * 0.5 = 2^-7 (which is subnormal in E4M3)
            subnormal_result = 0x01  # Represents smallest subnormal number
            yield from self.run_addition_test(
                0x01, 0x00, subnormal_result, "Subnormal result", compare_bits=True
            )

            # Test case 8: Exact boundary value where result needs to be normalized
            # 0.5 + 0.5 = 1.0 (requires normalization)
            yield from self.run_addition_test(
                0x30, 0x30, 0x38, "Normalization boundary"
            )

            # Test case 9: Rounding case
            # A case where rounding should occur
            # 3.0 + 0.75 = 3.75, which should round correctly
            yield from self.run_addition_test(0x44, 0x34, 0x47, "Rounding case")

            # Test case 10: NaN handling
            # Any operation with NaN should result in NaN
            nan_value = 0x7F  # NaN in E4M3 (01111111)
            # NaN + normal number = NaN
            yield from self.run_addition_test(
                nan_value, 0x40, nan_value, "NaN + normal number"
            )
            # NaN + NaN = NaN
            yield from self.run_addition_test(
                nan_value, nan_value, nan_value, "NaN + NaN"
            )

            # Test case 11: Addition that causes exponent overflow
            # Result should saturate to max value (not NaN)
            large_exp_a = 0x70  # 64.0
            large_exp_b = 0x70  # 64.0
            yield from self.run_addition_test(
                large_exp_a, large_exp_b, max_finite, "Exponent overflow"
            )

            # Test case 12: Subtraction that causes exponent underflow
            # Result should normalize to smallest representable value
            small_sub_a = 0x20  # 0.25
            small_sub_b = 0xA0  # -0.25
            yield from self.run_addition_test(
                small_sub_a, small_sub_b, 0x00, "Exponent underflow"
            )

            # Test case 13: Mixed sign addition with significant bit loss
            # Testing a case where subtracting close numbers causes significant bits to be lost
            # 2.0 - 1.96875 ≈ 0.03125
            yield from self.run_addition_test(
                0x40, 0xBF, 0x08, "Mixed sign with precision loss"
            )

            # Test case 14: Large value but not overflow
            # Test a large value that is within range
            yield from self.run_addition_test(0x7C, 0x40, 0x7C, "Large value addition")

        # Run simulation
        self.sim = test_runner(
            self.create_fp8_adder,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="fp8_e4m3_add_edge_cases",
            vcd_output=True,
            duration=1000,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
