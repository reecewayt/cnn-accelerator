from myhdl import *


@block
def mac(clk, reset, a, b, clear, result):
    """
    A simple multiply-accumulate (MAC) unit.

    Parameters:
    - clk: Clock signal
    - reset: Active-high reset signal
    - a, b: Input operands for multiplication
    - clear: Signal to clear the accumulator
    - result: Output result (accumulated value)
    """
    # Internal accumulator register
    acc = Signal(intbv(0, min=result.min, max=result.max))

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
    def accumulate():
        if clear:
            acc.next = 0
        else:
            # Perform multiplication and accumulation
            acc.next = acc + (a * b)

    # Connect internal accumulator to output
    @always_comb
    def output_logic():
        result.next = acc

    if not isinstance(reset, ResetSignal):
        return accumulate, output_logic, reset_connect
    else:
        return accumulate, output_logic
