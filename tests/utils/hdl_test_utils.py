import os
import inspect
import glob
from myhdl import *


def test_runner(
    dut_function,
    test_function,
    clk,
    period=10,
    dut_name=None,
    vcd_output=True,
    duration=None,
    *args,
    **kwargs,
):
    """
    Comprehensive test runner for MyHDL modules with optional VCD generation.
    Args:
        dut_function: Function that returns the device under test
        test_function: Generator function for testing
        clk: Optional clock signal to drive automatically
        period: Clock period if clock signal is provided (default: 10)
        dut_name: Optional name to use for the DUT in VCD filename
        vcd_output: Enable or disable VCD generation (default: True)
        duration: Optional simulation duration
        *args, **kwargs: Arguments to pass to dut_function
    Returns:
        The simulation results
    """
    if clk is None:
        raise ValueError("Clock signal must be provided for test_runner")
    if not isinstance(clk, SignalType):
        raise ValueError("Clock signal must be a MyHDL Signal")

    # Create the DUT instance
    dut_inst = dut_function(*args, **kwargs)

    # Handle VCD tracing if enabled
    if vcd_output:
        # Get the calling test name
        frame = inspect.currentframe().f_back
        caller_function = frame.f_code.co_name

        # Use provided DUT name or function name
        if dut_name is None:
            dut_name = dut_function.__name__
            # Remove 'create_' prefix if it exists
            if dut_name.startswith("create_"):
                dut_name = dut_name[7:]

        # Determine component name from the class name
        caller_class = frame.f_locals.get("self").__class__.__name__
        component_name = caller_class.replace("Test", "").lower()

        # Create component-specific directory
        component_dir = os.path.join("vcd", component_name)
        os.makedirs(component_dir, exist_ok=True)

        # Create VCD filename
        vcd_name = f"{dut_name}_{caller_function}"
        vcd_pattern = os.path.join(component_dir, f"{vcd_name}*.vcd")

        # Clean up old VCD files with the same name
        for old_file in glob.glob(vcd_pattern):
            os.remove(old_file)

        # Configure tracing
        traceSignals.directory = component_dir
        traceSignals.filename = vcd_name
        traced_dut = traceSignals(dut_inst)
        dut_for_sim = traced_dut
    else:
        # Use the DUT directly without tracing
        dut_for_sim = dut_inst

    # Get the test generator instance
    test_inst = test_function()

    # Create a list of instances for the simulation
    instances = [dut_for_sim, test_inst]
    clk_gen_inst = clock_gen(clk, period)
    instances.append(clk_gen_inst)

    # Create and run simulation
    sim = Simulation(*instances)

    # Run with optional duration
    if duration is not None:
        sim.run(duration=duration, quiet=1)
    else:
        sim.run(quiet=1)

    # Report VCD file creation if enabled
    if vcd_output:
        new_files = glob.glob(vcd_pattern)
        if new_files:
            print(f"\nCreated VCD file: {new_files[0]}")

    return sim


def clock_gen(clk, period=10):
    """
    Clock generator for MyHDL simulations.

    Args:
        clk: The clock signal to toggle.
        period: The period of the clock in time units (default is 10).
    """

    @always(delay(period // 2))
    def _clk_gen():
        clk.next = not clk

    return _clk_gen
