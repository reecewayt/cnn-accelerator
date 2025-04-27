import unittest

from myhdl import *
import sys
import os
import math

# Add the src directory to the path so we can import our module
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.hdl.components.mac import mac
from tests.utils.hdl_test_utils import simulate_with_vcd


class TestMacUnit(unittest.TestCase):
    """Test case for the MAC (Multiply-Accumulate) unit with VCD tracing."""

    def testBasicFunction(self):
        """Test basic MAC functionality with simple values."""
        # Define test parameters
        a_width = 8
        b_width = 8
        result_width = 32

        # Create signals
        clk = Signal(bool(0))
        reset = ResetSignal(0, active=1, isasync=False)
        a = Signal(intbv(0)[a_width:])
        b = Signal(intbv(0)[b_width:])
        enable = Signal(bool(0))
        result = Signal(
            intbv(0, min=-(2 ** (result_width - 1)), max=2 ** (result_width - 1))
        )
        overflow = Signal(bool(0))

        # Define the DUT creation function with a clear name
        def create_mac():
            return mac(clk, reset, a, b, enable, result, overflow)

        # Define generator function for test sequence
        @instance
        def test_sequence():
            # Initialize inputs
            a.next = 0
            b.next = 0
            enable.next = 0
            yield delay(10)

            # Test 1: Basic multiplication and accumulation
            a.next = 2
            b.next = 3
            enable.next = 1

            # Clock edge
            yield delay(5)
            clk.next = 1
            yield delay(5)
            clk.next = 0

            # Verify: 2 * 3 = 6 accumulated
            assert result == 6, f"Expected 6, got {result}"
            assert overflow == 0, f"Expected overflow=0, got {overflow}"

            # Test 2: Accumulate more values
            a.next = 4
            b.next = 5

            yield delay(5)
            clk.next = 1
            yield delay(5)
            clk.next = 0

            # Verify: 6 + (4 * 5) = 26 accumulated
            assert result == 26, f"Expected 26, got {result}"

            # Test reset
            reset.next = 1
            yield delay(5)
            clk.next = 1
            yield delay(5)
            clk.next = 0

            # Verify reset
            assert result == 0, f"Expected 0 after reset, got {result}"

        # Run simulation with VCD generation
        simulate_with_vcd(create_mac, lambda: test_sequence)

    def testOverflow(self):
        """Test overflow detection in the MAC unit."""
        # Define test parameters
        a_width = 8
        b_width = 8
        result_width = 16  # Smaller result to trigger overflow

        # Create signals
        clk = Signal(bool(0))
        reset = ResetSignal(0, active=1, isasync=False)
        a = Signal(intbv(0)[a_width:])
        b = Signal(intbv(0)[b_width:])
        enable = Signal(bool(0))
        result = Signal(intbv(0)[result_width:])
        overflow = Signal(bool(0))

        # Define the DUT creation function with a clear name
        def create_mac():
            return mac(clk, reset, a, b, enable, result, overflow)

        # Define generator function for test sequence
        @instance
        def test_sequence():
            # Initialize with reset
            val_a = 100  # constants
            val_b = 100  # constants

            reset.next = 1
            yield delay(5)
            clk.next = 1
            yield delay(5)
            clk.next = 0
            reset.next = 0

            # Set values close to overflow
            a.next = val_a
            b.next = val_b
            enable.next = 1

            # First accumulation
            yield delay(5)
            clk.next = 1
            yield delay(5)
            clk.next = 0

            # Should be 10000
            assert result == (val_a * val_b), f"Expected 10000, got {result}"
            assert overflow == 0, f"Expected overflow=0, got {overflow}"

            # Cycle until overflow should happen
            num_cycles = math.ceil((2**16) // (val_a * val_b))
            for _ in range(num_cycles):
                yield delay(5)
                clk.next = 1
                yield delay(5)
                clk.next = 0

            # Should set overflow flag and saturate
            assert overflow == 1, f"Expected overflow=1, got {overflow}"
            assert result == (2**16) - 1, f"Expected 65535 (saturated), got {result}"

        # Run simulation with VCD generation
        simulate_with_vcd(create_mac, lambda: test_sequence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
