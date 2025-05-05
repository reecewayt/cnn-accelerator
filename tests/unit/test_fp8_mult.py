import unittest
from myhdl import *
import sys
import os

# Import your module and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.hdl.components.fp8_e4m3_mult import fp8_e4m3_multiply
from src.utils.fp_defs import E4M3Format
from tests.utils.hdl_test_utils import test_runner


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

        self.sim = None

    def tearDown(self):
        """Clean up after each test."""
        if self.sim is not None:
            self.sim.quit()

    def create_fp8_multiplier(self):
        """Helper to create multiplier instance with current signals."""
        # Using only 5 parameters as per the implementation
        return fp8_e4m3_multiply(
            self.input_a, self.input_b, self.output_z, self.clk, self.rst
        )

    def fp8_to_float(self, fp8_val):
        """Convert an E4M3 value to Python float for verification."""
        # Extract components
        sign = (fp8_val >> 7) & 0x1
        exp = (fp8_val >> 3) & 0xF
        frac = fp8_val & 0x7

        # Handle special cases
        if exp == 0xF:  # All 1's in exponent
            if frac == 0:
                return float("-inf") if sign else float("inf")
            else:
                return float("nan")
        elif exp == 0 and frac == 0:
            return -0.0 if sign else 0.0

        # Normal or denormal processing
        if exp == 0:  # Denormal
            normalized_frac = frac / 8.0
            unbiased_exp = -6  # -7 + 1
        else:  # Normal
            normalized_frac = (frac / 8.0) + 1.0
            unbiased_exp = exp - 7  # Remove bias

        # Calculate value
        value = normalized_frac * (2.0**unbiased_exp)
        return -value if sign else value

    def float_to_fp8(self, f):
        """Convert a Python float to E4M3 format for testing."""
        if f == 0:
            return 0

        # Handle special values
        if f == float("inf"):
            return 0x78  # 0111.1000
        if f == float("-inf"):
            return 0xF8  # 1111.1000
        if f != f:  # NaN check
            return 0x7F  # 0111.1111

        # Handle regular values
        sign = 0x80 if f < 0 else 0
        f = abs(f)

        # Find exponent
        exp = 0
        while f >= 2.0:
            f /= 2.0
            exp += 1
        while f < 1.0 and exp > -7:
            f *= 2.0
            exp -= 1

        # Handle denormals
        if exp <= -7:
            # Denormal number
            frac = int(round(f * 8))
            return sign | frac

        # Normal numbers
        biased_exp = exp + 7
        frac = int(round((f - 1.0) * 8))

        # Handle overflow
        if biased_exp > 14:
            return sign | 0x78  # Infinity with sign

        return sign | (biased_exp << 3) | frac

    def testBasicMultiplication(self):
        """Test basic multiplication with two simple values."""

        @instance
        def test_sequence():
            # Reset the system
            self.rst.next = 1
            yield delay(20)
            self.rst.next = 0
            yield delay(10)

            # Test case: 1.5 * 2.0 = 3.0
            a_val = 1.5
            b_val = 2.0
            expected = 3.0

            # Convert float values to E4M3 format
            a_fp8 = self.float_to_fp8(a_val)
            b_fp8 = self.float_to_fp8(b_val)
            expected_fp8 = self.float_to_fp8(expected)

            # Print hex values for debugging
            print(f"Input A: {a_val} => 0x{a_fp8:02x}")
            print(f"Input B: {b_val} => 0x{b_fp8:02x}")
            print(f"Expected: {expected} => 0x{expected_fp8:02x}")

            # Set inputs
            self.input_a.next = a_fp8
            self.input_b.next = b_fp8

            # Wait for a few clock cycles for the result
            for _ in range(5):
                yield delay(10)
                self.clk.next = 1
                yield delay(10)
                self.clk.next = 0

            # Read result and convert to float
            result_float = self.fp8_to_float(int(self.output_z))

            print(f"Result: 0x{int(self.output_z):02x} => {result_float}")
            print(f"Test: {a_val} * {b_val} = {result_float}, Expected: {expected}")

            # Verify result (allow small difference due to rounding)
            assert (
                abs(result_float - expected) < 0.1
            ), f"Expected {expected}, got {result_float} for {a_val} * {b_val}"

            # Second test case: 0.5 * 4.0 = 2.0
            a_val = 0.5
            b_val = 4.0
            expected = 2.0

            # Convert float values to E4M3 format
            a_fp8 = self.float_to_fp8(a_val)
            b_fp8 = self.float_to_fp8(b_val)
            expected_fp8 = self.float_to_fp8(expected)

            # Print hex values for debugging
            print(f"Input A: {a_val} => 0x{a_fp8:02x}")
            print(f"Input B: {b_val} => 0x{b_fp8:02x}")
            print(f"Expected: {expected} => 0x{expected_fp8:02x}")

            # Set inputs
            self.input_a.next = a_fp8
            self.input_b.next = b_fp8

            # Wait for a few clock cycles for the result
            for _ in range(5):
                yield delay(10)
                self.clk.next = 1
                yield delay(10)
                self.clk.next = 0

            # Read result and convert to float
            result_float = self.fp8_to_float(int(self.output_z))

            print(f"Result: 0x{int(self.output_z):02x} => {result_float}")
            print(f"Test: {a_val} * {b_val} = {result_float}, Expected: {expected}")

            # Verify result (allow small difference due to rounding)
            assert (
                abs(result_float - expected) < 0.1
            ), f"Expected {expected}, got {result_float} for {a_val} * {b_val}"

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
