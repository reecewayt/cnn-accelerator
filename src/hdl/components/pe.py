from myhdl import *


@block
def processing_element(
    clk,
    i_reset,
    i_a,
    i_b,
    i_enable,
    i_clear,
    o_result,
    o_overflow,
    o_done,
    data_width=8,
    acc_width=32,
):
    # Constants
    acc_min = -(2 ** (acc_width - 1))
    acc_max = 2 ** (acc_width - 1) - 1
    prod_min = -(2 ** (2 * data_width - 1))
    prod_max = 2 ** (2 * data_width - 1) - 1

    # Internal signals
    accumulator = Signal(intbv(0, min=acc_min, max=acc_max + 1))
    product = Signal(intbv(0, min=prod_min, max=prod_max + 1))
    sum_result = Signal(intbv(0, min=2 * acc_min, max=2 * acc_max + 1))

    product_latched = Signal(intbv(0, min=prod_min, max=prod_max + 1))
    valid_product = Signal(bool(0))  # Marks when product_latched is valid

    overflow_flag = Signal(bool(0))
    done_flag = Signal(bool(0))

    # Combinational multiplication (Cycle 1)
    @always_comb
    def comb_product():
        product.next = i_a * i_b

    # Sequential MAC operation (Cycle 1 + Cycle 2)
    @always_seq(clk.posedge, reset=i_reset)
    def seq_logic():
        if i_clear or i_reset:
            accumulator.next = 0
            product_latched.next = 0
            o_result.next = 0
            valid_product.next = False
            done_flag.next = False
            overflow_flag.next = False
        elif i_enable and not valid_product:
            # Cycle 1: latch product
            product_latched.next = product
            valid_product.next = True
            done_flag.next = False
        elif valid_product:
            # Cycle 2: accumulate and check overflow
            temp_sum = int(accumulator) + int(product_latched)
            if temp_sum > acc_max:
                accumulator.next = acc_max
                overflow_flag.next = True
            elif temp_sum < acc_min:
                accumulator.next = acc_min
                overflow_flag.next = True
            else:
                accumulator.next = temp_sum
                overflow_flag.next = False

            done_flag.next = True
            valid_product.next = False
        else:
            done_flag.next = False

    # Output assignments
    @always_comb
    def comb_output():
        o_result.next = accumulator
        o_overflow.next = overflow_flag
        o_done.next = done_flag

    return instances()
