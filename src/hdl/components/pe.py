"""
Processing element

"""

from myhdl import *
from src.hdl.components.mac import mac
from src.hdl.components.reg import register


@block
def pe(
    clk,
    reset,
    a_in,
    b_in,
    a_out,
    b_out,
    c_out,
    read_result,
    data_width=32,
    acc_width=64,
):
    """
    A simple processing element (PE) that performs multiplication and accumulation.

    Parameters:
    - clk: Clock signal
    - reset: Active-high reset signal
    - a_in, b_in: Input operands for multiplication
    - a_out, b_out: Pass-through outputs for the inputs
    - c_out: Output result (accumulated value)
    - clear: Signal to clear the accumulator
    - data_width: Width of the input data
    - acc_width: Width of the accumulator
    """
    if not isinstance(reset, ResetSignal):
        raise ValueError("Reset signal must be a ResetSignal")

    mac_out = Signal(intbv(0)[acc_width:0])

    reg_a_inst = register(clk, reset, a_in, 1, a_out, width=data_width)
    reg_b_inst = register(clk, reset, b_in, 1, b_out, width=data_width)
    mac_inst = mac(clk, reset, a_in, b_in, reset, mac_out)

    @always_comb
    def output_logic():
        if read_result:
            c_out.next = mac_out

    return instances()
