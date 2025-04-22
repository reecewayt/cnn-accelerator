import unittest
from myhdl import *
import os
import inspect
import random
import sys
import glob


class HDLTestUtils(unittest.TestCase):
    """
    Base test case class for MyHDL modules that provides common functionality
    for simulation setup, VCD management, and signal monitoring.
    """

    # For debugging
    DEBUG = False

    def setUp(self):
        """Set up common signals and directories."""
        # Common signals
        self.clk = Signal(bool(0))
        self.reset = ResetSignal(1, active=1, isasync=False)

        # Create build directory if it doesn't exist
        os.makedirs("build", exist_ok=True)

    def simulate(self, test_bench, duration=1000, trace=True):
        """
        Run a simulation with test-specific VCD files.

        Args:
            test_bench: MyHDL block function to simulate
            duration: Simulation duration
            trace: Whether to generate VCD trace files

        Returns:
            The simulation instance
        """
        # Get the calling test method name for custom VCD filename
        caller_frame = inspect.currentframe().f_back
        caller_method = caller_frame.f_code.co_name

        if trace:
            # Clean up old VCD files for this test
            pattern = os.path.join("build", f"{caller_method}*.vcd")
            for old_file in glob.glob(pattern):
                os.remove(old_file)

            # Configure tracing
            traceSignals.directory = "build"
            traceSignals.filename = caller_method

        # Run simulation
        tb = test_bench()
        tb.config_sim(trace=trace)
        tb.run_sim(duration)

        if trace:
            # Find the new VCD file
            new_files = glob.glob(pattern)
            if new_files:
                print(f"\nCreated VCD file: {new_files[0]}")

        return tb

    def debug(self, message):
        """
        Print debug messages only when DEBUG is enabled.

        Args:
            message: Message to print when in debug mode
        """
        if self.__class__.DEBUG:
            print(message)

    def check_signal(self, signal, expected_value, error_msg=None):
        """
        Check if a signal has the expected value and generate a meaningful error message.

        Args:
            signal: MyHDL signal to check
            expected_value: Expected value
            error_msg: Optional custom error message prefix
        """
        if error_msg is None:
            error_msg = f"Signal check failed"

        signal_value = int(signal)
        if isinstance(expected_value, int) and expected_value > 255:
            # Use hex representation for large values
            msg = (
                f"{error_msg}: expected {hex(expected_value)}, got {hex(signal_value)}"
            )
        else:
            msg = f"{error_msg}: expected {expected_value}, got {signal_value}"

        self.assertEqual(signal_value, expected_value, msg)

    @classmethod
    def set_debug(cls, debug):
        """
        Set the debug mode for the test class.

        Args:
            debug: Boolean value to enable or disable debug mode
        """
        cls.DEBUG = debug

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests in the class."""
        # Additional cleanup could be added here if needed
        pass
