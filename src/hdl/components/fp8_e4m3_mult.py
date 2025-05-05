from myhdl import *
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.utils.fp_defs import E4M3Format


@block
def fp8_e4m3_multiply(input_a, input_b, output_z, clk, rst):
    """
    Simplified E4M3 floating-point multiplier
    Parameters:
    - input_a, input_b: Input E4M3 operands (8-bit each)
    - output_z: Output E4M3 product (8-bit)
    - clk, rst: Clock and reset signals
    """
    # Constants from E4M3Format
    WIDTH = E4M3Format.WIDTH  # 8
    EXP_BITS = E4M3Format.EXP_BITS  # 4
    MAN_BITS = E4M3Format.MAN_BITS  # 3
    EXP_BIAS = E4M3Format.EXP_BIAS  # 7

    # Input signals breakdown
    a_sign = Signal(bool(0))
    a_exp = Signal(intbv(0)[EXP_BITS:])
    a_man = Signal(intbv(0)[MAN_BITS:])

    b_sign = Signal(bool(0))
    b_exp = Signal(intbv(0)[EXP_BITS:])
    b_man = Signal(intbv(0)[MAN_BITS:])

    # Extended mantissa with implicit bit
    a_man_ext = Signal(intbv(0)[MAN_BITS + 1 :])
    b_man_ext = Signal(intbv(0)[MAN_BITS + 1 :])

    # Output signals
    z_sign = Signal(bool(0))
    z_exp = Signal(intbv(0)[EXP_BITS:])
    z_man = Signal(intbv(0)[MAN_BITS:])

    # Intermediate calculation signals
    exp_sum = Signal(intbv(0, min=-(2 * EXP_BIAS), max=2 * EXP_BIAS))
    product = Signal(intbv(0)[2 * (MAN_BITS + 1) :])
    normalized_exp = Signal(intbv(0)[EXP_BITS:])
    normalized_man = Signal(intbv(0)[MAN_BITS:])

    # Special case flags
    a_is_zero = Signal(bool(0))
    b_is_zero = Signal(bool(0))
    a_is_inf = Signal(bool(0))
    b_is_inf = Signal(bool(0))
    a_is_nan = Signal(bool(0))
    b_is_nan = Signal(bool(0))

    # Result flags
    result_is_nan = Signal(bool(0))
    result_is_inf = Signal(bool(0))
    result_is_zero = Signal(bool(0))

    @always_comb
    def decompose_inputs():
        # Extract components from input_a
        a_sign.next = bool(input_a[WIDTH - 1])
        a_exp.next = (input_a[WIDTH - 1 : MAN_BITS]) & ((1 << EXP_BITS) - 1)
        a_man.next = input_a[MAN_BITS:] & ((1 << MAN_BITS) - 1)

        # Extract components from input_b
        b_sign.next = bool(input_b[WIDTH - 1])
        b_exp.next = (input_b[WIDTH - 1 : MAN_BITS]) & ((1 << EXP_BITS) - 1)
        b_man.next = input_b[MAN_BITS:] & ((1 << MAN_BITS) - 1)

        # Add implicit '1' bit for normalized numbers
        a_man_ext.next = (1 << MAN_BITS) | a_man if a_exp != 0 else a_man
        b_man_ext.next = (1 << MAN_BITS) | b_man if b_exp != 0 else b_man

        # Detect special cases
        a_is_zero.next = (a_exp == 0) and (a_man == 0)
        b_is_zero.next = (b_exp == 0) and (b_man == 0)
        a_is_inf.next = (a_exp == (1 << EXP_BITS) - 1) and (a_man == 0)
        b_is_inf.next = (b_exp == (1 << EXP_BITS) - 1) and (b_man == 0)
        a_is_nan.next = (a_exp == (1 << EXP_BITS) - 1) and (a_man != 0)
        b_is_nan.next = (b_exp == (1 << EXP_BITS) - 1) and (b_man != 0)

    @always(clk.posedge)
    def multiply_proc():
        if rst:
            output_z.next = 0
        else:
            # Determine sign (XOR of input signs)
            z_sign.next = a_sign ^ b_sign

            # Check for special cases
            if a_is_nan or b_is_nan:
                # NaN * anything = NaN
                result_is_nan.next = True
                result_is_inf.next = False
                result_is_zero.next = False
            elif (a_is_inf and b_is_zero) or (b_is_inf and a_is_zero):
                # Inf * 0 = NaN
                result_is_nan.next = True
                result_is_inf.next = False
                result_is_zero.next = False
            elif a_is_inf or b_is_inf:
                # Inf * non-zero = Inf
                result_is_nan.next = False
                result_is_inf.next = True
                result_is_zero.next = False
            elif a_is_zero or b_is_zero:
                # 0 * anything = 0
                result_is_nan.next = False
                result_is_inf.next = False
                result_is_zero.next = True
            else:
                # Regular multiplication
                result_is_nan.next = False
                result_is_inf.next = False
                result_is_zero.next = False

                # Add exponents (subtract bias)
                if a_exp == 0:  # Denormal
                    a_adj_exp = -EXP_BIAS + 1
                else:
                    a_adj_exp = a_exp - EXP_BIAS

                if b_exp == 0:  # Denormal
                    b_adj_exp = -EXP_BIAS + 1
                else:
                    b_adj_exp = b_exp - EXP_BIAS

                exp_sum.next = a_adj_exp + b_adj_exp

                # Multiply mantissas
                product.next = a_man_ext * b_man_ext

                # Normalization
                if product[2 * (MAN_BITS + 1) - 1]:  # Check if highest bit is set
                    normalized_man.next = product[
                        2 * (MAN_BITS + 1) - 1 : 2 * (MAN_BITS + 1) - 1 - MAN_BITS
                    ]
                    exp_sum.next = exp_sum + 1
                else:
                    normalized_man.next = product[
                        2 * (MAN_BITS + 1) - 2 : 2 * (MAN_BITS + 1) - 2 - MAN_BITS
                    ]

                # Check for overflow or underflow
                if exp_sum >= EXP_BIAS:
                    # Overflow to infinity
                    result_is_inf.next = True
                elif exp_sum < -EXP_BIAS + 1:
                    # Underflow to zero
                    result_is_zero.next = True
                else:
                    # Normal result, adjust exponent
                    z_exp.next = exp_sum + EXP_BIAS
                    z_man.next = normalized_man

    @always_comb
    def assemble_output():
        if result_is_nan:
            # NaN representation: all 1s in exponent, non-zero mantissa
            output_z.next = ((1 << EXP_BITS) - 1) << MAN_BITS | (1 << (MAN_BITS - 1))
        elif result_is_inf:
            # Infinity: all 1s in exponent, zero mantissa, correct sign
            output_z.next = (z_sign << (WIDTH - 1)) | (
                ((1 << EXP_BITS) - 1) << MAN_BITS
            )
        elif result_is_zero:
            # Zero: just the sign bit
            output_z.next = z_sign << (WIDTH - 1)
        else:
            # Normal number
            output_z.next = (z_sign << (WIDTH - 1)) | (z_exp << MAN_BITS) | z_man

    return decompose_inputs, multiply_proc, assemble_output
