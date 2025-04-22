from myhdl import *


@block
def register(clk, reset, d, en, q, width=64):
    """
    A simple register with optional enable and reset functionality.
    Parameters:
    - clk: Clock signal
    - reset: Active-high reset signal
    - d: Data input
    - q: Data output
    - en: Enable signal (optional)
    - width: Width of the register (default is 64 bits)
    """

    # Storage for the register
    _reg = Signal(intbv(0)[width:0])

    # Check if reset is already a ResetSignal, if not, create one
    if not isinstance(reset, ResetSignal):
        reset_sig = ResetSignal(1, active=1, isasync=False)

        # Connect the reset input to our ResetSignal
        @always_comb
        def reset_connect():
            reset_sig.next = reset

    else:
        reset_sig = reset

    @always_seq(clk.posedge, reset=reset_sig)
    def reg_logic():
        if reset_sig:
            _reg.next = 0
        elif en:
            _reg.next = d

    @always_comb
    def output_logic():
        q.next = _reg

    # Return the actual generator functions
    if not isinstance(reset, ResetSignal):
        return reset_connect, reg_logic, output_logic
    else:
        return reg_logic, output_logic
