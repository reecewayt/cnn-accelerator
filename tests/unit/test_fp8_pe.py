"""
Test for the Floating Point Processing Element (FP8_PE)
"""

import unittest
from myhdl import *
import sys
import os

# Import your module and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.hdl.components.fp8_pe import fp8_pe
from tests.utils.hdl_test_utils import test_runner
from tests.utils.fp8_helpers import float_to_fp8, fp8_to_float


class TestFP8PE(unittest.TestCase):
    """Test case for the Floating Point Processing Element."""

    def setUp(self):
        # Test values with expected results
        self.test_values = [
            (1.5, 2.0, 3.0),  # 1.5 * 2.0 = 3.0
            (3.0, 1.5, 4.5),  # 3.0 * 1.5 = 4.5
            (-2.0, 1.5, -3.0),  # -2.0 * 1.5 = -3.0
            (0.5, 8.0, 4.0),  # 0.5 * 8.0 = 4.0
        ]

        # Setup
        self.sim = None
        self.data_width = 8  # 8 bits for E4M3 format

        # Signals
        self.clk = Signal(bool(0))
        self.reset = ResetSignal(0, active=1, isasync=False)

        # Inputs
        self.i_a = Signal(intbv(0)[self.data_width :])
        self.i_b = Signal(intbv(0)[self.data_width :])
        self.i_data_valid = Signal(bool(0))
        self.i_read_en = Signal(bool(0))
        self.i_clear_acc = Signal(bool(0))

        # Outputs
        self.o_c = Signal(intbv(0)[self.data_width :])
        self.o_mac_done = Signal(bool(0))
        self.o_ready_for_new = Signal(bool(0))

    def tearDown(self):
        if self.sim is not None:
            self.sim.quit()

    # DUT creation
    def create_fp8_pe(self):
        return fp8_pe(
            clk=self.clk,
            i_a=self.i_a,
            i_b=self.i_b,
            i_data_valid=self.i_data_valid,
            i_read_en=self.i_read_en,
            i_reset=self.reset,
            i_clear_acc=self.i_clear_acc,
            o_c=self.o_c,
            o_mac_done=self.o_mac_done,
            o_ready_for_new=self.o_ready_for_new,
            data_width=self.data_width,
        )

    def testBasicFunction(self):
        """Test basic FP8_PE functionality with simple values."""

        @instance
        def test_sequence():
            # Reset the system
            self.reset.next = 1
            yield self.clk.posedge
            self.reset.next = 0
            yield self.clk.posedge

            # Clear the accumulator
            self.i_clear_acc.next = 1
            yield self.clk.posedge
            self.i_clear_acc.next = 0
            yield self.clk.posedge

            # Test sequence for MAC operations
            cumulative_result = 0.0

            for a_val, b_val, expected in self.test_values:
                # Convert input values to E4M3 format
                a_fp8 = float_to_fp8(a_val)
                b_fp8 = float_to_fp8(b_val)

                print(f"\nMAC operation: {a_val} * {b_val}")
                print(f"A: {a_val} => 0x{a_fp8:02x}")
                print(f"B: {b_val} => 0x{b_fp8:02x}")

                # Wait for PE to be ready
                while not self.o_ready_for_new:
                    yield self.clk.posedge

                # Set inputs and start MAC
                self.i_a.next = a_fp8
                self.i_b.next = b_fp8
                self.i_data_valid.next = 1
                yield self.clk.posedge
                self.i_data_valid.next = 0

                # Wait for MAC operation to complete
                while not self.o_mac_done:
                    yield self.clk.posedge

                # Accumulate expected result (for verification)
                cumulative_result += expected

                # Read result
                self.i_read_en.next = 1
                yield self.clk.posedge
                self.i_read_en.next = 0
                yield self.clk.posedge

                # Get and verify result
                result_fp8 = int(self.o_c)
                result_float = fp8_to_float(result_fp8)

                print(f"Expected: {cumulative_result}")
                print(f"Result: {result_float} (0x{result_fp8:02x})")

                # Check that result is close to expected (allowing for FP rounding)
                self.assertAlmostEqual(
                    result_float,
                    cumulative_result,
                    delta=0.5,
                    msg=f"Expected ~{cumulative_result}, got {result_float}",
                )

            # Test clear accumulator
            self.i_clear_acc.next = 1
            yield self.clk.posedge
            self.i_clear_acc.next = 0
            yield self.clk.posedge

            # Read cleared accumulator
            self.i_read_en.next = 1
            yield self.clk.posedge
            self.i_read_en.next = 0
            yield self.clk.posedge

            # Verify it's zero
            result_fp8 = int(self.o_c)
            result_float = fp8_to_float(result_fp8)
            print(f"\nAfter clear: {result_float} (0x{result_fp8:02x})")
            self.assertEqual(
                result_fp8, 0, f"Expected 0x00 after clear, got 0x{result_fp8:02x}"
            )

        # Run simulation
        self.sim = test_runner(
            self.create_fp8_pe,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="fp8_pe",
            vcd_output=True,
            verilog_output=True,
            duration=2000,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
