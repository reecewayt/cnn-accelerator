from myhdl import *
import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.hdl.components.pe import pe
from tests.utils.hdl_test_utils import HDLTestUtils


class TestPE(HDLTestUtils):
    """
    Test case for the Processing Element (PE) module with wave-based control.
    """

    def test_pe_basic_operation(self):
        """
        Test basic operation of the PE module.
        """
        # Set up test signals
        data_width = 8  # Using smaller width for simpler testing
        acc_width = 16

        # Data signals - use consistent slice notation [width:0]
        i_a = Signal(intbv(0)[data_width:0])
        i_b = Signal(intbv(0)[data_width:0])
        o_a = Signal(intbv(0)[data_width:0])
        o_b = Signal(intbv(0)[data_width:0])
        o_c = Signal(intbv(0)[acc_width:0])

        # Control signals
        i_en = Signal(bool(0))
        i_read_result = Signal(bool(0))

        @block
        def test_bench():
            # Instantiate the PE module with the correct parameters
            dut = pe(
                self.clk,
                self.reset,
                i_en,
                i_a,
                i_b,
                o_a,
                o_b,
                o_c,
                i_read_result,
                data_width,
                acc_width,
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
                i_a.next = 3
                i_b.next = 5
                i_en.next = 1  # Enable PE
                yield self.clk.posedge
                self.debug(f"a_out = {o_a}, b_out = {o_b}, mac_out = {int(o_c)}")

                # Keep enabled for another cycle to propagate
                yield self.clk.posedge

                # Check a_out and b_out signals - should propagate after one clock cycle
                self.check_signal(o_a, 3, "a_out after first cycle")
                self.check_signal(o_b, 5, "b_out after first cycle")

                # Test case 2: Another multiplication to test accumulation
                i_a.next = 2
                i_b.next = 4
                yield self.clk.posedge
                self.debug(f"a_out = {o_a}, b_out = {o_b}, mac_out = {int(o_c)}")

                # Keep enabled for another cycle
                yield self.clk.posedge

                # Check register propagation again
                self.check_signal(o_a, 2, "a_out after second cycle")
                self.check_signal(o_b, 4, "b_out after second cycle")

                # Enable read_result to see accumulated value
                i_read_result.next = 1
                yield self.clk.posedge
                self.debug(f"read_result = {i_read_result}, c_out = {int(o_c)}")

                # Check result - should be 3*5 + 2*4 = 15 + 8 = 23
                self.check_signal(o_c, 23, "c_out after two multiplications")

                # Run a few more cycles to flush the pipeline
                i_en.next = 0  # Disable PE
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

        # Data signals - use consistent slice notation [width:0]
        i_a = Signal(intbv(0)[data_width:0])
        i_b = Signal(intbv(0)[data_width:0])
        o_a = Signal(intbv(0)[data_width:0])
        o_b = Signal(intbv(0)[data_width:0])
        o_c = Signal(intbv(0)[acc_width:0])

        # Control signals
        i_en = Signal(bool(0))
        i_read_result = Signal(bool(0))

        @block
        def test_bench():
            # Instantiate the PE module
            dut = pe(
                self.clk,
                self.reset,
                i_en,
                i_a,
                i_b,
                o_a,
                o_b,
                o_c,
                i_read_result,
                data_width,
                acc_width,
            )

            # Clock generator
            @always(delay(5))
            def clk_gen():
                self.clk.next = not self.clk

            # Stimulus generator
            @instance
            def stimulus():
                # Set initial values
                i_a.next = 5
                i_b.next = 6
                i_en.next = 1  # Enable PE

                # Reset the PE
                self.reset.next = 1
                yield self.clk.posedge
                yield self.clk.posedge

                # Check registers are reset
                self.check_signal(o_a, 0, "a_out should be reset to 0")
                self.check_signal(o_b, 0, "b_out should be reset to 0")

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
                i_read_result.next = 1
                yield self.clk.posedge

                # Check that MAC was reset
                self.check_signal(o_c, 0, "c_out should be reset to 0")

                # End simulation
                raise StopSimulation

            return instances()

        # Run the simulation
        self.simulate(test_bench, duration=200, trace=True)

    def test_pe_wave_propagation(self):
        """
        Test wave-based propagation through the PE.
        """
        # Set up test signals
        data_width = 8
        acc_width = 16

        # Data signals - use consistent slice notation [width:0]
        i_a = Signal(intbv(0)[data_width:0])
        i_b = Signal(intbv(0)[data_width:0])
        o_a = Signal(intbv(0)[data_width:0])
        o_b = Signal(intbv(0)[data_width:0])
        o_c = Signal(intbv(0)[acc_width:0])

        # Control signals
        i_en = Signal(bool(0))
        i_read_result = Signal(bool(0))

        @block
        def test_bench():
            # Instantiate the PE module
            dut = pe(
                self.clk,
                self.reset,
                i_en,
                i_a,
                i_b,
                o_a,
                o_b,
                o_c,
                i_read_result,
                data_width,
                acc_width,
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

                # Test wave-based propagation by pulsing enable signal
                # First wave
                i_a.next = 1
                i_b.next = 2
                i_en.next = 1
                yield self.clk.posedge
                i_en.next = 0
                yield self.clk.posedge

                # Second wave
                i_a.next = 3
                i_b.next = 4
                i_en.next = 1
                yield self.clk.posedge
                i_en.next = 0
                yield self.clk.posedge

                # Third wave
                i_a.next = 5
                i_b.next = 6
                i_en.next = 1
                yield self.clk.posedge
                i_en.next = 0
                yield self.clk.posedge

                # Read accumulated result
                i_read_result.next = 1
                i_en.next = 1  # Need enable for read_result to take effect
                yield self.clk.posedge

                # Result should be 1*2 + 3*4 + 5*6 = 2 + 12 + 30 = 44
                self.check_signal(o_c, 44, "c_out after three waves")

                # End simulation
                raise StopSimulation

            return instances()

        # Run the simulation
        self.simulate(test_bench, duration=200, trace=True)


if __name__ == "__main__":
    unittest.main()
