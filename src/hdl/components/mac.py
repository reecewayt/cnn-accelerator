from myhdl import *


@block
def mac(clk, reset, a, b, enable, result):
    """
    A simple multiply-accumulate (MAC) unit with single reset functionality.

    Parameters:
    - clk: Clock signal
    - reset: Active-high reset signal (clears accumulator)
    - a, b: Input operands for multiplication
    - enable: Enable signal for accumulation
    - result: Output result (accumulated value)
    """
    # Internal accumulator register
    acc = Signal(intbv(0, min=result.min, max=result.max))

    # Ensure reset is a proper ResetSignal
    if not isinstance(reset, ResetSignal):
        reset_sig = ResetSignal(1, active=1, isasync=False)

        @always_comb
        def reset_connect():
            reset_sig.next = reset

    else:
        reset_sig = reset

    @always_seq(clk.posedge, reset=reset_sig)
    def accumulate():
        if enable:
            # Perform multiplication and accumulation when enabled
            acc.next = acc + (a * b)

    # Connect internal accumulator to output
    @always_comb
    def output_logic():
        result.next = acc

    if not isinstance(reset, ResetSignal):
        return accumulate, output_logic, reset_connect
    else:
        return accumulate, output_logic
