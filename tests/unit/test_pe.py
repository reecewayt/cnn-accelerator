import unittest
from myhdl import *
import numpy as np
import sys
import os

# Import your module and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.hdl.components.pe import processing_element
from tests.utils.hdl_test_utils import test_runner


class TestProcessingElement(unittest.TestCase):
    """Test case for the Processing Element module."""

    def setUp(self):
        """Set up common signals and parameters for the tests."""
        self.sim = None
        self.data_width = 8
        self.acc_width = 32

        self.clk = Signal(bool(0))

        # FIXED: Use simple bit width specifications to match PE expectations
        self.i_a = Signal(intbv(0)[self.data_width :])
        self.i_b = Signal(intbv(0)[self.data_width :])

        self.i_enable = Signal(bool(0))
        self.i_clear = Signal(bool(0))
        self.i_reset = ResetSignal(0, active=1, isasync=False)

        # FIXED: Use simple bit width for output
        self.o_result = Signal(intbv(0)[self.acc_width :])
        self.o_overflow = Signal(bool(0))
        self.o_done = Signal(bool(0))

    def tearDown(self):
        if self.sim is not None:
            self.sim.quit()

    def create_processing_element(self):
        """Create the processing element instance."""
        return processing_element(
            clk=self.clk,
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

            # Set inputs - FIXED: Handle signed values properly
            self.i_a.next = 5
            self.i_b.next = 3
            self.i_enable.next = 1
            yield self.clk.posedge
            self.i_enable.next = 0

            # Wait for done signal
            while not self.o_done:
                yield self.clk.posedge

            # Check result - FIXED: Use .signed() for proper comparison
            print(f"Result: {self.o_result.signed()}, Expected: 15")
            self.assertEqual(self.o_result.signed(), 15)
            self.assertFalse(self.o_overflow)

            # Clear the accumulator
            self.i_clear.next = 1
            yield self.clk.posedge
            self.i_clear.next = 0
            yield self.clk.posedge

            # Check accumulator is cleared
            self.assertEqual(self.o_result, 0)
            print("Basic test passed!")

        self.sim = test_runner(
            self.create_processing_element,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="processing_element",
            vcd_output=True,
            verilog_output=True,
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

            # FIXED: Handle negative values properly with intbv
            self.i_a.next = intbv(-3)[self.data_width :]  # -3 as 8-bit signed
            self.i_b.next = 4
            self.i_enable.next = 1
            yield self.clk.posedge
            self.i_enable.next = 0

            while not self.o_done:
                yield self.clk.posedge

            print(f"Negative test result: {self.o_result.signed()}, Expected: -12")
            self.assertEqual(self.o_result.signed(), -12)
            self.assertFalse(self.o_overflow)
            print("Negative test passed!")

        self.sim = test_runner(
            self.create_processing_element,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="processing_element_negative",
            vcd_output=True,
            verilog_output=True,
            duration=1000,
        )


if __name__ == "__main__":
    unittest.main()
