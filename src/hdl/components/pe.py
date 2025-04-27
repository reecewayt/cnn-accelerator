"""
Processing element with wave-based control for systolic array
"""

from myhdl import *
from src.hdl.components.mac import mac


@block
def pe(
    # Clock and control
    i_clk,
    i_reset,
    i_en,  # Enable signal (wave control)
    # Data signals
    i_a,  # A input operand
    i_b,  # B input operand
    o_a,  # A output (pass-through)
    o_b,  # B output (pass-through)
    o_c,  # C output (result)
    i_read_result,  # Read accumulated result
    data_width=32,
    acc_width=64,
):
    """
    A processing element (PE) with wave-based control for systolic arrays.

    Parameters:
    - i_clk: Clock signal input
    - i_reset: Active-high reset signal input (clears all registers)
    - i_en: Enable signal for wave propagation
    - i_a, i_b: Input operands for multiplication
    - o_a, o_b: Pass-through outputs for the inputs
    - o_c: Output result (accumulated value)
    - i_read_result: Signal to read the accumulated result
    - data_width: Width of the input data
    - acc_width: Width of the accumulator
    """
    if not isinstance(i_reset, ResetSignal):
        raise ValueError("Reset signal must be a ResetSignal")

    # Internal signals
    mac_out = Signal(intbv(0)[acc_width:0])

    # Internal registers (must be sequential)
    a_reg = Signal(intbv(0)[data_width:0])
    b_reg = Signal(intbv(0)[data_width:0])
    c_reg = Signal(intbv(0)[acc_width:0])

    # MAC enable signal
    mac_enable = Signal(bool(0))

    # MAC instance with simplified interface
    mac_inst = mac(i_clk, i_reset, a_reg, b_reg, mac_enable, mac_out)

    @always_comb
    def control_logic():
        # MAC is enabled when PE is enabled
        mac_enable.next = i_en

    @always_seq(i_clk.posedge, reset=i_reset)
    def register_logic():
        if i_en:
            # Update data registers when enabled
            a_reg.next = i_a
            b_reg.next = i_b

            # Update result register when read_result is high
            if i_read_result:
                c_reg.next = mac_out

    @always_comb
    def output_logic():
        # Connect registers to outputs (always, not conditionally)
        o_a.next = a_reg
        o_b.next = b_reg
        o_c.next = c_reg

    return instances()
