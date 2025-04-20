from myhdl import *
import random
import sys
import os

# Add the src directory to the path so we can import our module
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.hdl.components.mac import mac


@block
def mac_tb():
    # Define signals with appropriate bit widths
    clk = Signal(bool(0))
    reset = ResetSignal(
        1, active=1, isasync=False
    )  # Use ResetSignal instead of regular Signal
    a = Signal(intbv(0, min=-128, max=127))  # 8-bit signed
    b = Signal(intbv(0, min=-128, max=127))  # 8-bit signed
    clear = Signal(bool(0))
    result = Signal(
        intbv(0, min=-(2**16), max=2**16 - 1)
    )  # 16-bit for accumulated results

    # Instantiate the MAC
    dut = mac(clk, reset, a, b, clear, result)

    # Clock generator
    @always(delay(10))
    def clkgen():
        clk.next = not clk

    # Stimulus and checking
    @instance
    def stimulus():
        # Initialize and reset
        reset.next = 1
        yield clk.posedge
        reset.next = 0

        # Test case 1: Basic accumulation
        for i in range(5):
            a.next = i
            b.next = 2
            yield clk.posedge
            print(f"a={int(a)}, b={int(b)}, result={int(result)}")

        # Test case 2: Clear accumulator
        clear.next = 1
        yield clk.posedge
        clear.next = 0

        # Test case 3: More accumulation after clear
        for i in range(3):
            a.next = i + 10
            b.next = 3
            yield clk.posedge
            print(f"a={int(a)}, b={int(b)}, result={int(result)}")

        # End simulation
        raise StopSimulation

    return dut, clkgen, stimulus


# This allows the file to be run directly
if __name__ == "__main__":
    # Create directory for output files
    os.makedirs("build", exist_ok=True)

    # Specify the VCD file path
    vcd_path = os.path.join("build", "mac_tb.vcd")

    # Run the simulation
    tb = mac_tb()

    # Set the traceSignals.name attribute to control VCD file name
    traceSignals.directory = "build"
    traceSignals.name = "mac_tb"

    tb.config_sim(trace=True)
    tb.run_sim()
