import unittest
from myhdl import *
import sys
import os
import math

# Import your module and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.hdl.components.mac import mac
from tests.utils.hdl_test_utils import simulate_with_vcd
from tests.utils.hdl_test_utils import test_runner


class TestMacUnit(unittest.TestCase):
    """Test case for the MAC (Multiply-Accumulate) unit with VCD tracing."""

    def setUp(self):
        """Setup common signals and parameters for all tests."""
        # Common parameters
        self.a_width = 8
        self.b_width = 8

        # Common signals
        self.clk = Signal(bool(0))
        self.reset = ResetSignal(0, active=1, isasync=False)
        self.a = Signal(intbv(0)[self.a_width :])
        self.b = Signal(intbv(0)[self.b_width :])
        self.enable = Signal(bool(0))

        # These will be set in the test methods since result width varies
        self.result = None
        self.overflow = Signal(bool(0))

    def create_mac(self):
        """Helper to create MAC instance with current signals."""
        return mac(
            self.clk,
            self.reset,
            self.a,
            self.b,
            self.enable,
            self.result,
            self.overflow,
        )

    def clock_cycle(self):
        """Helper method for a clock cycle."""
        yield delay(5)
        self.clk.next = 1
        yield delay(5)
        self.clk.next = 0

    def testBasicFunction(self):
        """Test basic MAC functionality with simple values."""
        # Setup specific to this test
        result_width = 32
        self.result = Signal(
            intbv(0, min=-(2 ** (result_width - 1)), max=2 ** (result_width - 1))
        )

        @instance
        def test_sequence():
            # Initialize inputs
            self.a.next = 0
            self.b.next = 0
            self.enable.next = 0
            yield delay(10)

            # Test 1: Basic multiplication and accumulation
            self.a.next = 2
            self.b.next = 3
            self.enable.next = 1

            # Apply clock cycle using helper
            yield from self.clock_cycle()

            # Verify: 2 * 3 = 6 accumulated
            assert self.result == 6, f"Expected 6, got {self.result}"
            assert self.overflow == 0, f"Expected overflow=0, got {self.overflow}"

            # Additional test steps...
            self.a.next = 4
            self.b.next = 5
            yield from self.clock_cycle()

            # Verify: 6 + (4 * 5) = 26 accumulated
            assert self.result == 26, f"Expected 26, got {self.result}"

            # Test reset
            self.reset.next = 1
            yield from self.clock_cycle()

            # Verify reset
            assert self.result == 0, f"Expected 0 after reset, got {self.result}"

        # Run simulation with VCD generation
        simulate_with_vcd(self.create_mac, lambda: test_sequence, dut_name="mac")

    def testOverflow(self):
        """Test overflow detection in the MAC unit."""
        # Setup specific to this test
        result_width = 16  # Smaller result to trigger overflow
        self.result = Signal(intbv(0)[result_width:])

        @instance
        def test_sequence():
            val_a = 100  # constants
            val_b = 100  # constants

            # Initialize with reset
            self.reset.next = 1
            yield from self.clock_cycle()
            self.reset.next = 0

            # Set values close to overflow
            self.a.next = val_a
            self.b.next = val_b
            self.enable.next = 1

            # First accumulation
            yield from self.clock_cycle()

            # Verification steps...
            assert self.result == (val_a * val_b), f"Expected 10000, got {self.result}"
            assert self.overflow == 0, f"Expected overflow=0, got {self.overflow}"

            # Cycle until overflow should happen
            num_cycles = math.ceil((2**16) / (val_a * val_b))
            for _ in range(num_cycles):
                yield from self.clock_cycle()

            # Should set overflow flag and saturate
            assert self.overflow == 1, f"Expected overflow=1, got {self.overflow}"
            assert (
                self.result == (2**16) - 1
            ), f"Expected 65535 (saturated), got {self.result}"

        # Run simulation with VCD generation
        simulate_with_vcd(self.create_mac, lambda: test_sequence, dut_name="mac")

    def testResetFunctionality(self):
        result_width = 32
        self.result = Signal(intbv(0)[result_width:])

        val_a = 2
        val_b = 3

        @instance
        def test_sequence():
            # Reset the DUT - no internal clock generator needed
            self.reset.next = 1
            yield delay(10)  # Wait for a cycle
            self.reset.next = 0

            # Set inputs
            self.a.next = val_a
            self.b.next = val_b
            self.enable.next = 1

            # Wait for 20 cycles - use fixed delay to synchronize with the external clock
            for _ in range(20):
                # Wait for a clock cycle
                yield delay(10)

                # Alternatively, you can use:
                # yield from self.clk.posedge
                # yield delay(1)  # Small delay after posedge

            # Verify the result - should be val_a * val_b * 20
            self.assertEqual(
                self.result,
                (val_a * val_b) * 20,
                f"Expected {(val_a * val_b) * 20}, got {self.result}",
            )

        # Run test with automatic clock generation
        test_runner(
            self.create_mac,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="mac",
            vcd_output=True,
            duration=500,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
