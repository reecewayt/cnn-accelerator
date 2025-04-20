"""
Simple script to convert the MAC module to Verilog using MyHDL.
"""

from myhdl import *
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

# Import your MAC module
from src.hdl.components.mac import mac


def convert_mac_to_verilog():
    """Convert the MAC module to Verilog."""
    # Create signals with the same types used in your testbench
    clk = Signal(bool(0))
    reset = ResetSignal(1, active=1, isasync=False)
    a = Signal(intbv(0, min=-128, max=127))  # 8-bit signed
    b = Signal(intbv(0, min=-128, max=127))  # 8-bit signed
    clear = Signal(bool(0))
    result = Signal(
        intbv(0, min=-(2**16), max=2**16 - 1)
    )  # 16-bit for accumulated results

    # Create an instance of your module
    dut = mac(clk, reset, a, b, clear, result)

    # Make sure the output directory exists
    os.makedirs("gen/verilog", exist_ok=True)

    # Convert the instance to Verilog
    dut.convert(hdl="Verilog", path="gen/verilog", name="mac")

    print(f"Verilog code generated: gen/verilog/mac.v")


if __name__ == "__main__":
    convert_mac_to_verilog()
