import unittest
from myhdl import *
import numpy as np
import sys
import os

# Import your module and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.hdl.components.pe import processing_element
from tests.utils.hdl_test_utils import test_runner


class TestProcessingElementUnit(unittest.TestCase):
    """Test case for the Processing Element module."""

    def setUp(self):
        """Set up common signals and parameters for the tests."""
        # Parameters
        self.sim = None
        self.data_width = 8
        self.acc_width = 32

        # Add the missing clock signal
        self.clk = Signal(bool(0))

        # Define test inputs
        self.i_a = Signal(
            intbv(0, min=-(2 ** (self.data_width - 1)), max=2 ** (self.data_width - 1))
        )
        self.i_b = Signal(
            intbv(0, min=-(2 ** (self.data_width - 1)), max=2 ** (self.data_width - 1))
        )

        self.i_enable = Signal(bool(0))
        self.i_clear = Signal(bool(0))
        self.i_reset = ResetSignal(0, active=1, isasync=False)

        # Output signals
        self.o_result = Signal(
            intbv(0, min=-(2 ** (self.acc_width - 1)), max=2 ** (self.acc_width - 1))
        )

        self.o_overflow = Signal(bool(0))
        self.o_done = Signal(bool(0))  # Add done signal for timing control

    def tearDown(self):
        if self.sim is not None:
            self.sim.quit()

    def create_processing_element(self):
        """Create the processing element instance."""
        return processing_element(
            clk=self.clk,  # Use self.clk instead of Signal(bool(0))
            i_reset=self.i_reset,
            i_a=self.i_a,
            i_b=self.i_b,
            i_enable=self.i_enable,
            i_clear=self.i_clear,
            o_result=self.o_result,
            o_overflow=self.o_overflow,
            o_done=self.o_done,
            data_width=self.data_width,
            acc_width=self.acc_width,
        )

    def test_processing_element_basic(self):
        """Test basic functionality of the processing element."""

        @instance
        def test_sequence():
            # Reset the PE
            self.i_reset.next = 1
            yield self.clk.posedge
            self.i_reset.next = 0
            yield self.clk.posedge

            # Set inputs
            self.i_a.next = 5
            self.i_b.next = 3
            self.i_enable.next = 1
            self.i_clear.next = 0
            yield self.clk.posedge
            self.i_enable.next = 0  # Disable after one cycle
            while not self.o_done:
                yield self.clk.posedge

            # Check result
            self.assertEqual(self.o_result, 15)
            self.assertFalse(self.o_overflow)

            # Clear the accumulator
            self.i_clear.next = 1
            yield self.clk.posedge
            yield self.clk.negedge
            self.i_clear.next = 0
            yield self.clk.posedge

            # Check accumulator is cleared
            self.assertEqual(self.o_result, 0)

        # Move this inside the test method and fix the function calls
        self.sim = test_runner(
            self.create_processing_element,  # Remove parentheses
            lambda: test_sequence,  # Remove parentheses
            clk=self.clk,
            period=10,
            dut_name="processing_element",
            vcd_output=True,
            verilog_output=True,
            duration=500,
        )

    def test_overflow_behavior(self):
        """Test that overflow is detected and accumulator saturates properly."""

        @instance
        def test_sequence():
            self.i_reset.next = 1
            yield self.clk.posedge
            self.i_reset.next = 0
            yield self.clk.posedge

            # Use large positive values to cause overflow
            self.i_a.next = 2 ** (self.data_width - 1) - 1  # Max positive
            self.i_b.next = 2 ** (self.data_width - 1) - 1

            # Wait for overflow
            cycles = 0
            while not self.o_overflow:
                yield self.clk.posedge
                self.i_enable.next = 1
                yield self.clk.posedge
                cycles += 1
                self.i_enable.next = 0
                while not self.o_done:
                    yield self.clk.posedge
                    cycles = cycles + 1

            print(f"Overflow detected after {cycles} cycles")

            # Should be saturated at acc_max
            acc_max = 2 ** (self.acc_width - 1) - 1
            self.assertEqual(self.o_result, acc_max)
            self.assertTrue(self.o_overflow)

        self.sim = test_runner(
            self.create_processing_element,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="processing_element_overflow",
            vcd_output=True,
            duration=10000000,
        )

    def test_multi_cycle_accumulate(self):
        """Test accumulation over multiple MAC operations."""

        @instance
        def test_sequence():
            self.i_reset.next = 1
            yield self.clk.posedge
            self.i_reset.next = 0
            yield self.clk.posedge

            total = 0
            for i in range(5):
                self.i_a.next = i + 1
                self.i_b.next = 2
                self.i_enable.next = 1
                yield self.clk.posedge
                self.i_enable.next = 0
                while not self.o_done:
                    yield self.clk.posedge
                total += (i + 1) * 2
                self.assertEqual(self.o_result, total)

        self.sim = test_runner(
            self.create_processing_element,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="processing_element_accumulate",
            vcd_output=True,
            duration=1000,
        )

    def test_negative_multiplication(self):
        """Test handling of negative values in inputs."""

        @instance
        def test_sequence():
            self.i_reset.next = 1
            yield self.clk.posedge
            self.i_reset.next = 0
            yield self.clk.posedge

            self.i_a.next = -3
            self.i_b.next = 4
            self.i_enable.next = 1
            yield self.clk.posedge
            self.i_enable.next = 0
            while not self.o_done:
                yield self.clk.posedge

            self.assertEqual(self.o_result.signed(), -12)
            self.assertFalse(self.o_overflow)

        self.sim = test_runner(
            self.create_processing_element,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="processing_element_negative",
            vcd_output=True,
            duration=500,
        )


if __name__ == "__main__":
    unittest.main()
