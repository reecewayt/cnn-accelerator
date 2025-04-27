from myhdl import *


@block
def mac(clk, reset, a, b, enable, result, overflow):
    """
    A simple multiply-accumulate (MAC) unit for integer arithmetic. Future iteration
    of design should include floating point support.

    Parameters:
    - clk: Clock signal
    - reset: Active-high reset signal (clears accumulator)
    - a, b: Input operands for multiplication
    - enable: Enable signal for accumulation
    - clear: Signal to clear the accumulator without reset
    - result: Output result (accumulated value)
    - overflow: Overflow flag
    """
    # Determine bit widths
    a_width = len(a)
    b_width = len(b)
    result_width = len(result)

    # a plus b width to avoid overflow
    product = Signal(intbv(0)[a_width + b_width :])

    # Internal accumulator register with extra bit for overflow detection
    acc = Signal(intbv(0, min=result.min, max=result.max))

    if not isinstance(reset, ResetSignal):
        raise ValueError("Reset signal must be a ResetSignal")

    @always_comb
    def multiply():
        product.next = a * b

    @always_seq(clk.posedge, reset=reset)
    def accumulate():
        if reset:
            acc.next = 0
            overflow.next = False
        elif enable:
            # Calculate next value
            next_val = acc + product

            # Check for overflow
            if next_val >= result.max or next_val < result.min:
                overflow.next = True
                # Saturate output to max/min value if overflow occurs
                if next_val >= result.max:
                    acc.next = result.max - 1
                else:
                    acc.next = result.min
            else:
                overflow.next = False
                acc.next = next_val

    # Connect internal accumulator to output
    @always_comb
    def output_logic():
        result.next = acc

    return instances()
