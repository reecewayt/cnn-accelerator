"""
Processing element with simultaneous input for systolic array
"""

from myhdl import *
from src.hdl.components.mac import mac


@block
def pe(
    clk,
    # Inputs
    i_a,
    i_b,
    i_data_valid,
    i_read_en,
    i_reset,
    # Outputs
    o_a,
    o_b,
    o_c,
    o_saturate_detect,
    # Parameters
    data_width=32,
    acc_width=64,
):
    """
    Processing Element (PE) for systolic array architecture
    Elements see input values simultaneously, latching them when data is valid

    Parameters:
    - clk: Clock signal
    - i_a, i_b: Input operands
    - i_data_valid: Signal indicating valid input data
    - i_read_en: Enable reading of result
    - i_reset: Reset signal
    - o_a, o_b: Pass-through outputs
    - o_c: Output result
    - o_saturate_detect: Overflow/saturation detection flag
    - data_width: Width of input data
    - acc_width: Width of accumulator
    """
    # Reset signal check
    if not isinstance(i_reset, ResetSignal):
        raise ValueError("Reset signal must be a ResetSignal")

    # Internal signals for MAC unit
    mac_result = Signal(intbv(0)[acc_width:])
    mac_enable = Signal(bool(0))
    mac_overflow = Signal(bool(0))

    # Instantiate the MAC unit
    mac_unit = mac(
        clk=clk,
        reset=i_reset,
        a=i_a,
        b=i_b,
        enable=mac_enable,
        result=mac_result,
        overflow=mac_overflow,
    )

    @always_comb
    def mac_enable_logic():
        """
        Disable MAC unit if saturation is detected
        """
        if i_data_valid and not mac_overflow:
            mac_enable.next = True
        else:
            mac_enable.next = False

    @always_comb
    def output_logic():
        # Direct pass-through of input values
        o_a.next = i_a
        o_b.next = i_b

        # Output the MAC result when read is enabled
        if i_read_en:
            o_c.next = mac_result

        elif not i_read_en:
            o_c.next = 0

        # Connect saturation detect to MAC overflow
        o_saturate_detect.next = mac_overflow

    return instances()
