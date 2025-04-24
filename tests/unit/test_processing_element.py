from myhdl import *
import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.hdl.components.pe import pe
from tests.utils.hdl_test_utils import HDLTestUtils


class TestPE(HDLTestUtils):
    """
    Test case for the Processing Element (PE) module.
    """

    def test_pe_basic_operation(self):
        """
        Test basic operation of the PE module.
        """
        # Set up test signals
        data_width = 8  # Using smaller width for simpler testing
        acc_width = 16

        a_in = Signal(intbv(0)[data_width:])
        b_in = Signal(intbv(0)[data_width:])
        a_out = Signal(intbv(0)[data_width:])
        b_out = Signal(intbv(0)[data_width:])
        c_out = Signal(intbv(0)[acc_width:])
        read_result = Signal(bool(0))

        @block
        def test_bench():
            # Instantiate the PE module
            dut = pe(
                self.clk,
                self.reset,
                a_in,
                b_in,
                a_out,
                b_out,
                c_out,
                read_result,
                data_width=data_width,
                acc_width=acc_width,
            )

            # Clock generator
            @always(delay(5))
            def clk_gen():
                self.clk.next = not self.clk

            # Stimulus generator
            @instance
            def stimulus():
                # Reset the PE
                self.reset.next = 1
                yield self.clk.posedge
                self.reset.next = 0
                yield self.clk.posedge

                # Test case 1: Simple multiplication and accumulation
                a_in.next = 3
                b_in.next = 5
                yield self.clk.posedge
                self.debug(f"a_out = {a_out}, b_out = {b_out}, mac_out = {int(c_out)}")

                yield self.clk.posedge

                # Check a_out and b_out signals - should propagate after one clock cycle
                self.check_signal(a_out, 3, "a_out after first cycle")
                self.check_signal(b_out, 5, "b_out after first cycle")

                # Test case 2: Another multiplication to test accumulation
                yield self.clk.posedge
                a_in.next = 2
                b_in.next = 4
                yield self.clk.posedge
                self.debug(f"a_out = {a_out}, b_out = {b_out}, mac_out = {int(c_out)}")
                yield self.clk.posedge
                # Check register propagation again
                self.check_signal(a_out, 2, "a_out after second cycle")
                self.check_signal(b_out, 4, "b_out after second cycle")

                # Enable read_result to see accumulated value
                read_result.next = 1
                yield self.clk.posedge
                self.debug(f"read_result = {read_result}, c_out = {int(c_out)}")

                # Check result - should be 3*5 + 2*4 = 15 + 8 = 23
                # Note: Due to pipelining and timing, you may need to adjust this check based on your PE implementation
                self.check_signal(c_out, 23, "c_out after two multiplications")

                # Run a few more cycles to flush the pipeline
                yield self.clk.posedge
                yield self.clk.posedge

                # End simulation
                raise StopSimulation

            return instances()

        # Run the simulation
        self.simulate(test_bench, duration=200, trace=True)

    def test_pe_reset_behavior(self):
        """
        Test reset behavior of the PE module.
        """
        # Set up test signals
        data_width = 8
        acc_width = 16

        a_in = Signal(intbv(0)[data_width:])
        b_in = Signal(intbv(0)[data_width:])
        a_out = Signal(intbv(0)[data_width:])
        b_out = Signal(intbv(0)[data_width:])
        c_out = Signal(intbv(0)[acc_width:])
        read_result = Signal(bool(0))

        @block
        def test_bench():
            # Instantiate the PE module
            dut = pe(
                self.clk,
                self.reset,
                a_in,
                b_in,
                a_out,
                b_out,
                c_out,
                read_result,
                data_width=data_width,
                acc_width=acc_width,
            )

            # Clock generator
            @always(delay(5))
            def clk_gen():
                self.clk.next = not self.clk

            # Stimulus generator
            @instance
            def stimulus():
                # Set initial values
                a_in.next = 5
                b_in.next = 6

                # Reset the PE
                self.reset.next = 1
                yield self.clk.posedge
                yield self.clk.posedge

                # Check registers are reset
                self.check_signal(a_out, 0, "a_out should be reset to 0")
                self.check_signal(b_out, 0, "b_out should be reset to 0")

                # Release reset and perform operations
                self.reset.next = 0
                yield self.clk.posedge

                # Accumulate some values
                yield self.clk.posedge
                yield self.clk.posedge

                # Reset again mid-operation
                self.reset.next = 1
                yield self.clk.posedge

                # Enable read_result
                read_result.next = 1
                yield self.clk.posedge

                # Check that MAC was reset
                self.check_signal(c_out, 0, "c_out should be reset to 0")

                # End simulation
                raise StopSimulation

            return instances()

        # Run the simulation
        self.simulate(test_bench, duration=200, trace=True)


if __name__ == "__main__":
    unittest.main()
