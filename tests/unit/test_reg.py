import unittest
from myhdl import *
import os
import random
import sys
import inspect

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.hdl.components.reg import register
from tests.utils.hdl_test_utils import HDLTestUtils


class RegisterTestCase(HDLTestUtils):
    """Test case class specifically for register modules."""

    def create_register_test(self, width, test_values=None):
        """
        Create a test bench for a register with the specified width.

        Args:
            width: Bit width of the register
            test_values: Optional list of test values to use

        Returns:
            Test bench function
        """
        if test_values is None:
            # Default test values based on width
            test_values = [
                0xA5,  # 10100101 pattern
                (2**width) - 1,  # All 1s
                0,  # All 0s
                random.randint(0, 2**width - 1),  # Random value
            ]

        @block
        def test_bench():
            # Create signals
            d = Signal(intbv(0)[width:0])
            q = Signal(intbv(0)[width:0])
            en = Signal(bool(1))

            # Instantiate register
            dut = register(self.clk, self.reset, d, en, q, width=width)

            # Define clock generator directly here
            @always(delay(5))
            def clkgen():
                self.clk.next = not self.clk

            # Test sequence
            @instance
            def stimulus():
                self.reset.next = 1
                yield self.clk.posedge
                self.reset.next = 0
                yield self.clk.posedge

                # Test case 1: Basic write operation
                self.debug(f"\nTest {width}-bit register: Basic write")
                for value in test_values:
                    d.next = value
                    en.next = 1
                    yield self.clk.posedge
                    yield self.clk.posedge  # Extra cycle
                    yield delay(1)
                    self.debug(
                        f"  Setting d={hex(int(d))}, en=1, Result: q={hex(int(q))}"
                    )
                    self.check_signal(q, value, f"{width}-bit register write")

                # Test case 2: Disable operation
                self.debug(f"\nTest {width}-bit register: Disable")
                last_value = int(q)
                en.next = 0
                yield self.clk.posedge
                d.next = (last_value + 1) % (2**width)  # Different value
                yield self.clk.posedge
                yield delay(1)
                self.debug(f"  Setting d={hex(int(d))}, en=0, Result: q={hex(int(q))}")
                self.check_signal(q, last_value, f"{width}-bit register disable")

                # Test case 3: Reset operation
                self.debug(f"\nTest {width}-bit register: Reset")
                en.next = 1
                self.reset.next = 1
                yield self.clk.posedge
                yield delay(1)
                self.debug(f"  Setting reset=1, Result: q={hex(int(q))}")
                self.check_signal(q, 0, f"{width}-bit register reset")

                print(f"\n{width}-bit register tests complete")
                raise StopSimulation

            return dut, clkgen, stimulus

        return test_bench

    def test_16bit_register(self):
        """Test 16-bit register."""
        test_bench = self.create_register_test(16)
        self.simulate(test_bench)

    def test_32bit_register(self):
        """Test 32-bit register."""
        test_bench = self.create_register_test(32)
        self.simulate(test_bench)

    def test_64bit_register(self):
        """Test 64-bit register."""
        test_bench = self.create_register_test(64)
        self.simulate(test_bench)
