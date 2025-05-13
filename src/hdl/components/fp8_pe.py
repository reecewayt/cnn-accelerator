"""
Floating Point Processing Element with E4M3 MAC unit for systolic array
"""

from myhdl import *
from src.hdl.components.fp8_mac import fp8_e4m3_mac


@block
def fp8_pe(
    clk,
    # Inputs
    i_a,
    i_b,
    i_data_valid,
    i_read_en,
    i_reset,
    i_clear_acc,  # New signal to clear accumulator
    # Outputs
    o_c,
    o_mac_done,
    o_ready_for_new,
    # Parameters
    data_width=8,
):
    """
    Floating Point Processing Element (FP_PE) using E4M3 format for systolic array architecture.
    Elements see input values simultaneously, latching them when data is valid.

    Parameters:
    - clk: Clock signal
    - i_a, i_b: Input operands (E4M3 format)
    - i_data_valid: Signal indicating valid input data
    - i_read_en: Enable reading of result
    - i_reset: Reset signal
    - i_clear_acc: Signal to clear the accumulator
    - o_c: Output result (E4M3 format)
    - o_mac_done: Signal indicating MAC operation is complete
    - o_ready_for_new: Signal indicating PE is ready for new inputs
    """
    # Reset signal check
    if not isinstance(i_reset, ResetSignal):
        raise ValueError("Reset signal must be a ResetSignal")

    # Internal signals for controlling the MAC
    mac_start = Signal(bool(0))
    mac_done = Signal(bool(0))
    mac_ready = Signal(bool(0))

    # Output register
    output_reg = Signal(intbv(0)[data_width:])

    # Instantiate the MAC unit
    mac_unit = fp8_e4m3_mac(
        clk=clk,
        rst=i_reset,
        input_a=i_a,
        input_b=i_b,
        mac_start=mac_start,
        clear_acc=i_clear_acc,
        read_enable=i_read_en,
        output_result=output_reg,
        mac_done=mac_done,
        ready_for_new=mac_ready,
    )

    @always_comb
    def control_logic():
        """
        Control logic for the MAC unit
        """
        # Start the MAC when data is valid and MAC is ready
        mac_start.next = i_data_valid and mac_ready

        # Forward MAC status signals to outputs
        o_mac_done.next = mac_done
        o_ready_for_new.next = mac_ready

    @always_comb
    def output_logic():
        """
        Connect the MAC output to the PE output
        """
        o_c.next = output_reg

    return instances()
