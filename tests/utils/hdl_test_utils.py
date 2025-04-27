import unittest
from myhdl import *
import os
import inspect
import random
import sys
import glob


def simulate_with_vcd(dut_function, test_function, dut_name=None, *args, **kwargs):
    """
    Helper function to run a MyHDL simulation with VCD generation.
    Places VCD files in the vcd directory with proper naming.

    Args:
        dut_function: Function that returns the device under test
        test_function: Generator function for testing
        dut_name: Optional name to use for the DUT in VCD filename (defaults to function name)
        *args, **kwargs: Arguments to pass to dut_function

    Returns:
        The simulation results
    """
    # Create directory for VCD files if it doesn't exist
    os.makedirs("vcd", exist_ok=True)

    # Get the calling test name
    frame = inspect.currentframe().f_back
    caller_function = frame.f_code.co_name

    # Use provided DUT name or function name
    if dut_name is None:
        dut_name = dut_function.__name__
        # Remove 'create_' prefix if it exists
        if dut_name.startswith("create_"):
            dut_name = dut_name[7:]

    # Create VCD filename
    vcd_name = f"{dut_name}_{caller_function}"
    vcd_pattern = os.path.join("vcd", f"{vcd_name}*.vcd")

    # Clean up old VCD files with the same name
    for old_file in glob.glob(vcd_pattern):
        os.remove(old_file)

    # Create the traced DUT instance
    dut_inst = dut_function(*args, **kwargs)

    # Configure tracing
    traceSignals.directory = "vcd"
    traceSignals.filename = vcd_name
    dut_traced = traceSignals(dut_inst)

    # Get the test generator instance
    test_inst = test_function()

    # Create and run simulation
    sim = Simulation(dut_traced, test_inst)
    sim.run(quiet=1)

    # Report VCD file creation
    new_files = glob.glob(vcd_pattern)
    if new_files:
        print(f"\nCreated VCD file: {new_files[0]}")

    return sim
