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

            # Test case 1: 1.5 + 2.0 = 3.5
            a_val = 1.5
            b_val = 2.0
            expected = 3.5

            # Convert float values to E4M3 format
            a_fp8 = float_to_fp8(a_val)
            b_fp8 = float_to_fp8(b_val)
            expected_fp8 = float_to_fp8(expected)

            # Print hex values for debugging
            print(f"Test 1: {a_val} + {b_val} = {expected}")
            print(f"Input A: {a_val} => 0x{a_fp8:02x}")
            print(f"Input B: {b_val} => 0x{b_fp8:02x}")
            print(f"Expected: {expected} => 0x{expected_fp8:02x}")

            # Set inputs and start computation
            self.input_a.next = a_fp8
            self.input_b.next = b_fp8
            self.start.next = 1
            yield self.clk.posedge
            self.start.next = 0  # Start pulse for one clock cycle

            # Wait for computation to complete
            while not self.done:
                yield self.clk.posedge

            # Read result and convert to float
            result_float = int(self.output_z)

            print(f"Result: 0x{int(self.output_z):02x} => {result_float}")

            # Verify result (allow small difference due to rounding)
            assert (
                result_float == expected_fp8
            ), f"Expected {expected_fp8}, got {result_float} for {a_val} + {b_val}"

            # Wait an extra cycle between tests
            yield self.clk.posedge

            # Test case 2: 2.0 + (-1.5) = 0.5
            a_val = 2.0
            b_val = -1.5
            expected = 0.5

            # Convert float values to E4M3 format
            a_fp8 = float_to_fp8(a_val)
            b_fp8 = float_to_fp8(b_val)
            expected_fp8 = float_to_fp8(expected)

            # Print hex values for debugging
            print(f"\nTest 2: {a_val} + {b_val} = {expected}")
            print(f"Input A: {a_val} => 0x{a_fp8:02x}")
            print(f"Input B: {b_val} => 0x{b_fp8:02x}")
            print(f"Expected: {expected} => 0x{expected_fp8:02x}")

            # Set inputs and start computation
            self.input_a.next = a_fp8
            self.input_b.next = b_fp8
            self.start.next = 1
            yield self.clk.posedge
            self.start.next = 0  # Start pulse for one clock cycle

            # Wait for computation to complete
            while not self.done:
                yield self.clk.posedge

            # Read result and convert to float
            result_float = int(self.output_z)

            print(f"Result: 0x{int(self.output_z):02x} => {result_float}")

            # Verify result
            assert (
                result_float == expected_fp8
            ), f"Expected {expected}, got {result_float} for {a_val} + {b_val}"

            # Wait an extra cycle between tests
            yield self.clk.posedge

            # Test case 3: Add numbers that require alignment
            a_val = 4.0
            b_val = 0.25
            expected = 4.25

            # Convert float values to E4M3 format
            a_fp8 = float_to_fp8(a_val)
            b_fp8 = float_to_fp8(b_val)
            expected_fp8 = float_to_fp8(expected)

            # Print hex values for debugging
            print(f"\nTest 3: {a_val} + {b_val} = {expected}")
            print(f"Input A: {a_val} => 0x{a_fp8:02x}")
            print(f"Input B: {b_val} => 0x{b_fp8:02x}")
            print(f"Expected: {expected} => 0x{expected_fp8:02x}")

            # Set inputs and start computation
            self.input_a.next = a_fp8
            self.input_b.next = b_fp8
            self.start.next = 1
            yield self.clk.posedge
            self.start.next = 0  # Start pulse for one clock cycle

            # Wait for computation to complete
            while not self.done:
                yield self.clk.posedge

            # Read result and convert to float
            result_float = int(self.output_z)

            print(f"Result: 0x{int(self.output_z):02x} => {result_float}")

            # Verify result
            assert (
                result_float == expected_fp8
            ), f"Expected {expected}, got {result_float} for {a_val} + {b_val}"

            # Test case 4: Test zero handling
            print("\nTest 4: Zero handling tests")

            # 4.1: 0.0 + 2.0 = 2.0
            a_val = 0.0
            b_val = 2.0
            expected = 2.0

            a_fp8 = float_to_fp8(a_val)
            b_fp8 = float_to_fp8(b_val)
            expected_fp8 = float_to_fp8(expected)

            print(f"Test 4.1: {a_val} + {b_val} = {expected}")
            print(f"Input A: {a_val} => 0x{a_fp8:02x}")
            print(f"Input B: {b_val} => 0x{b_fp8:02x}")
            print(f"Expected: {expected} => 0x{expected_fp8:02x}")

            # Set inputs and start computation
            self.input_a.next = a_fp8
            self.input_b.next = b_fp8
            self.start.next = 1
            yield self.clk.posedge
            self.start.next = 0

            # Wait for computation to complete
            while not self.done:
                yield self.clk.posedge

            # Read result and convert to float
            result_float = fp8_to_float(int(self.output_z))
            print(f"Result: 0x{int(self.output_z):02x} => {result_float}")

            # Verify result
            assert (
                abs(result_float - expected) < 0.1
            ), f"Expected {expected}, got {result_float} for {a_val} + {b_val}"

            # Wait an extra cycle between tests
            yield self.clk.posedge

            # 4.2: 3.0 + 0.0 = 3.0
            a_val = 3.0
            b_val = 0.0
            expected = 3.0

            a_fp8 = float_to_fp8(a_val)
            b_fp8 = float_to_fp8(b_val)
            expected_fp8 = float_to_fp8(expected)

            print(f"\nTest 4.2: {a_val} + {b_val} = {expected}")
            print(f"Input A: {a_val} => 0x{a_fp8:02x}")
            print(f"Input B: {b_val} => 0x{b_fp8:02x}")
            print(f"Expected: {expected} => 0x{expected_fp8:02x}")

            # Set inputs and start computation
            self.input_a.next = a_fp8
            self.input_b.next = b_fp8
            self.start.next = 1
            yield self.clk.posedge
            self.start.next = 0

            # Wait for computation to complete
            while not self.done:
                yield self.clk.posedge

            # Read result and convert to float
            result_float = fp8_to_float(int(self.output_z))
            print(f"Result: 0x{int(self.output_z):02x} => {result_float}")

            # Verify result
            assert (
                abs(result_float - expected) < 0.1
            ), f"Expected {expected}, got {result_float} for {a_val} + {b_val}"

            # Wait an extra cycle between tests
            yield self.clk.posedge

            # 4.3: 0.0 + 0.0 = 0.0
            a_val = 0.0
            b_val = 0.0
            expected = 0.0

            a_fp8 = float_to_fp8(a_val)
            b_fp8 = float_to_fp8(b_val)
            expected_fp8 = float_to_fp8(expected)

            print(f"\nTest 4.3: {a_val} + {b_val} = {expected}")
            print(f"Input A: {a_val} => 0x{a_fp8:02x}")
            print(f"Input B: {b_val} => 0x{b_fp8:02x}")
            print(f"Expected: {expected} => 0x{expected_fp8:02x}")

            # Set inputs and start computation
            self.input_a.next = a_fp8
            self.input_b.next = b_fp8
            self.start.next = 1
            yield self.clk.posedge
            self.start.next = 0

            # Wait for computation to complete
            while not self.done:
                yield self.clk.posedge

            # Read result and convert to float
            result_float = fp8_to_float(int(self.output_z))
            print(f"Result: 0x{int(self.output_z):02x} => {result_float}")

            # Verify result
            assert (
                abs(result_float - expected) < 0.1
            ), f"Expected {expected}, got {result_float} for {a_val} + {b_val}"

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
