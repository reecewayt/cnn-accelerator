import unittest
from myhdl import *
import sys
import os

# Import your module and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.hdl.components.fp8_mac import fp8_e4m3_mac
from src.utils.fp_defs import E4M3Format
from tests.utils.hdl_test_utils import test_runner
from tests.utils.fp8_helpers import float_to_fp8, fp8_to_float


class TestFP8E4M3MAC(unittest.TestCase):
    """Test case for the pipelined E4M3 floating-point MAC unit."""

    def setUp(self):
        """Setup common signals and parameters for all tests."""
        # Common signals
        self.clk = Signal(bool(0))
        self.rst = ResetSignal(0, active=1, isasync=False)
        self.input_a = Signal(intbv(0)[E4M3Format.WIDTH :])
        self.input_b = Signal(intbv(0)[E4M3Format.WIDTH :])
        self.mac_start = Signal(bool(0))
        self.clear_acc = Signal(bool(0))
        self.read_enable = Signal(bool(0))
        self.output_result = Signal(intbv(0)[E4M3Format.WIDTH :])
        self.mac_done = Signal(bool(0))
        self.ready_for_new = Signal(bool(0))
        self.sim = None

    def tearDown(self):
        """Clean up after each test."""
        if self.sim is not None:
            self.sim.quit()

    def create_fp8_mac(self):
        """Helper to create MAC instance with current signals."""
        return fp8_e4m3_mac(
            self.clk,
            self.rst,
            self.input_a,
            self.input_b,
            self.mac_start,
            self.clear_acc,
            self.read_enable,
            self.output_result,
            self.mac_done,
            self.ready_for_new,
        )

    def testBasicMAC(self):
        """Test basic MAC operations - multiply and accumulate twice."""

        @instance
        def test_sequence():
            # Reset the system
            self.rst.next = 1
            yield self.clk.posedge
            yield self.clk.posedge
            self.rst.next = 0
            yield self.clk.posedge

            # Test case 1: Simple MAC operation
            # First clear the accumulator
            print("\n=== Test 1: Simple MAC Operation ===")
            print("Clearing accumulator...")
            self.clear_acc.next = 1
            yield self.clk.posedge
            self.clear_acc.next = 0
            yield self.clk.posedge

            # MAC operation 1: 2.0 * 3.0 = 6.0
            a1_val = 2.0
            b1_val = 3.0
            a1_fp8 = float_to_fp8(a1_val)
            b1_fp8 = float_to_fp8(b1_val)

            print(f"\nFirst MAC: {a1_val} * {b1_val}")
            print(f"Input A: {a1_val} => 0x{a1_fp8:02x}")
            print(f"Input B: {b1_val} => 0x{b1_fp8:02x}")

            # Wait for MAC to be ready
            while not self.ready_for_new:
                yield self.clk.posedge

            # Start first MAC operation
            self.input_a.next = a1_fp8
            self.input_b.next = b1_fp8
            self.mac_start.next = 1
            yield self.clk.posedge
            self.mac_start.next = 0

            # Wait for the first MAC to complete
            while not self.mac_done:
                yield self.clk.posedge
            print("First MAC done")

            # MAC operation 2: 1.5 * 4.0 = 6.0
            # Accumulator should become: 6.0 + 6.0 = 12.0
            a2_val = 1.5
            b2_val = 4.0
            a2_fp8 = float_to_fp8(a2_val)
            b2_fp8 = float_to_fp8(b2_val)

            print(f"\nSecond MAC: {a2_val} * {b2_val}")
            print(f"Input A: {a2_val} => 0x{a2_fp8:02x}")
            print(f"Input B: {b2_val} => 0x{b2_fp8:02x}")

            # Wait for MAC to be ready for new inputs
            while not self.ready_for_new:
                yield self.clk.posedge

            # Start second MAC operation
            self.input_a.next = a2_fp8
            self.input_b.next = b2_fp8
            self.mac_start.next = 1
            yield self.clk.posedge
            self.mac_start.next = 0

            # Wait for the second MAC to complete
            while not self.mac_done:
                yield self.clk.posedge
            print("Second MAC done")

            # Read the accumulated result
            self.read_enable.next = 1
            yield self.clk.posedge
            self.read_enable.next = 0
            yield self.clk.posedge

            # Get and verify result
            result_fp8 = int(self.output_result)
            result_float = fp8_to_float(result_fp8)
            expected_result = 12.0  # 6.0 + 6.0
            expected_fp8 = float_to_fp8(expected_result)

            print(f"\nFinal Result: 0x{result_fp8:02x} => {result_float}")
            print(f"Expected: 0x{expected_fp8:02x} => {expected_result}")

            # Allow for some floating-point error
            assert (
                abs(result_float - expected_result) < 0.5
            ), f"Expected {expected_result}, got {result_float}"

            # Test case 2: Test clear and new accumulation
            print("\n=== Test 2: Clear and New Accumulation ===")

            # Clear accumulator
            self.clear_acc.next = 1
            yield self.clk.posedge
            self.clear_acc.next = 0
            yield self.clk.posedge

            # Single MAC operation: -2.0 * 2.5 = -5.0
            a3_val = -2.0
            b3_val = 2.5
            a3_fp8 = float_to_fp8(a3_val)
            b3_fp8 = float_to_fp8(b3_val)

            print(f"\nMAC after clear: {a3_val} * {b3_val}")
            print(f"Input A: {a3_val} => 0x{a3_fp8:02x}")
            print(f"Input B: {b3_val} => 0x{b3_fp8:02x}")

            # Wait for ready
            while not self.ready_for_new:
                yield self.clk.posedge

            # Start MAC operation
            self.input_a.next = a3_fp8
            self.input_b.next = b3_fp8
            self.mac_start.next = 1
            yield self.clk.posedge
            self.mac_start.next = 0

            # Wait for completion
            while not self.mac_done:
                yield self.clk.posedge

            # Read result
            self.read_enable.next = 1
            yield self.clk.posedge
            self.read_enable.next = 0
            yield self.clk.posedge

            # Get and verify result
            result_fp8 = int(self.output_result)
            result_float = fp8_to_float(result_fp8)
            expected_result = -5.0
            expected_fp8 = float_to_fp8(expected_result)

            print(f"\nFinal Result: 0x{result_fp8:02x} => {result_float}")
            print(f"Expected: 0x{expected_fp8:02x} => {expected_result}")

            assert (
                abs(result_float - expected_result) < 0.5
            ), f"Expected {expected_result}, got {result_float}"

            print("\nAll tests passed!")

        # Run simulation
        self.sim = test_runner(
            self.create_fp8_mac,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="fp8_e4m3_mac",
            vcd_output=True,
            verilog_output=True,
            duration=2000,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
