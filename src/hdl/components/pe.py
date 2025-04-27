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
    """
    if not isinstance(i_reset, ResetSignal):
        raise ValueError("Reset signal must be a ResetSignal")

    # Internal registers
    a_reg = Signal(intbv(0)[data_width:0])
    b_reg = Signal(intbv(0)[data_width:0])
    c_reg = Signal(intbv(0)[acc_width:0])

    # Internal accumulator (required for combinational calculations)
    accumulator = Signal(intbv(0)[acc_width:0])

    # Temporary product signal
    product = Signal(intbv(0)[acc_width:0])

    # Combinational product calculation
    @always_comb
    def mult_logic():
        product.next = a_reg * b_reg

    # Sequential accumulation - c_reg must be updated in sequential logic
    @always_seq(i_clk.posedge, reset=i_reset)
    def register_logic():
        if i_reset:
            # Reset accumulator and registers
            a_reg.next = 0
            b_reg.next = 0
            accumulator.next = 0
        else:
            accumulator.next = accumulator + product
            if i_en:
                # Update data registers when enabled
                a_reg.next = i_a
                b_reg.next = i_b
                # Accumulate product in the same cycle
                accumulator.next = accumulator + product

            # Update result register when read_result is high
            if i_read_result:
                c_reg.next = accumulator

    # Output logic
    @always_comb
    def output_logic():
        # Connect registers to outputs (always, not conditionally)
        o_a.next = a_reg
        o_b.next = b_reg
        o_c.next = c_reg

    return instances()
