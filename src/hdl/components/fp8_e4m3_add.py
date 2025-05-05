from myhdl import *
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.utils.fp_defs import E4M3Format


@block
def fp8_e4m3_add(input_a, input_b, output_z, clk, rst):
    """
    E4M3 floating-point adder (combinational implementation)
    Parameters:
    - input_a, input_b: Input E4M3 operands (8-bit each)
    - output_z: Output E4M3 sum (8-bit)
    - clk, rst: Clock and reset signals (used only for registering output)
    """
    # Constants from E4M3Format
    WIDTH = E4M3Format.WIDTH  # 8
    EXP_BITS = E4M3Format.EXP_BITS  # 4
    MAN_BITS = E4M3Format.MAN_BITS  # 3
    EXP_BIAS = E4M3Format.EXP_BIAS  # 7

    # Internal signals for combinational logic
    result = Signal(intbv(0)[WIDTH:])

    # Input signals breakdown
    a_sign = Signal(bool(0))
    a_exp = Signal(intbv(0)[EXP_BITS:])
    a_man = Signal(intbv(0)[MAN_BITS:])

    b_sign = Signal(bool(0))
    b_exp = Signal(intbv(0)[EXP_BITS:])
    b_man = Signal(intbv(0)[MAN_BITS:])

    # Special case flags
    a_is_zero = Signal(bool(0))
    b_is_zero = Signal(bool(0))
    a_is_inf = Signal(bool(0))
    b_is_inf = Signal(bool(0))
    a_is_nan = Signal(bool(0))
    b_is_nan = Signal(bool(0))

    # Calculation signals
    a_man_ext = Signal(intbv(0)[MAN_BITS + 5 :])
    b_man_ext = Signal(intbv(0)[MAN_BITS + 5 :])
    larger_exp = Signal(intbv(0)[EXP_BITS:])
    exp_diff = Signal(intbv(0)[EXP_BITS:])
    aligned_a = Signal(intbv(0)[MAN_BITS + 5 :])
    aligned_b = Signal(intbv(0)[MAN_BITS + 5 :])
    add_result = Signal(intbv(0)[MAN_BITS + 6 :])
    final_sign = Signal(bool(0))
    final_exp = Signal(intbv(0)[EXP_BITS:])
    final_man = Signal(intbv(0)[MAN_BITS:])

    @always_comb
    def extract_components():
        # Extract components from inputs
        a_sign.next = bool(input_a[WIDTH - 1])
        a_exp.next = (input_a[WIDTH - 1 : MAN_BITS]) & ((1 << EXP_BITS) - 1)
        a_man.next = input_a[MAN_BITS:] & ((1 << MAN_BITS) - 1)

        b_sign.next = bool(input_b[WIDTH - 1])
        b_exp.next = (input_b[WIDTH - 1 : MAN_BITS]) & ((1 << EXP_BITS) - 1)
        b_man.next = input_b[MAN_BITS:] & ((1 << MAN_BITS) - 1)

        # Check for special cases
        a_is_zero.next = (a_exp == 0) and (a_man == 0)
        b_is_zero.next = (b_exp == 0) and (b_man == 0)
        a_is_inf.next = (a_exp == (1 << EXP_BITS) - 1) and (a_man == 0)
        b_is_inf.next = (b_exp == (1 << EXP_BITS) - 1) and (b_man == 0)
        a_is_nan.next = (a_exp == (1 << EXP_BITS) - 1) and (a_man != 0)
        b_is_nan.next = (b_exp == (1 << EXP_BITS) - 1) and (b_man != 0)

        # Extended mantissas with implicit bit
        if a_exp == 0:  # Denormal
            a_man_ext.next = a_man << 2
        else:
            a_man_ext.next = ((1 << MAN_BITS) | a_man) << 2

        if b_exp == 0:  # Denormal
            b_man_ext.next = b_man << 2
        else:
            b_man_ext.next = ((1 << MAN_BITS) | b_man) << 2

    @always_comb
    def align_mantissas():
        # Find larger exponent
        if a_exp >= b_exp:
            larger_exp.next = a_exp
            exp_diff.next = a_exp - b_exp

            # Align b's mantissa
            if b_exp == 0 and a_exp == 0:
                aligned_a.next = a_man_ext
                aligned_b.next = b_man_ext
            elif exp_diff >= MAN_BITS + 5:
                aligned_a.next = a_man_ext
                aligned_b.next = 0
            else:
                aligned_a.next = a_man_ext
                aligned_b.next = b_man_ext >> exp_diff
        else:
            larger_exp.next = b_exp
            exp_diff.next = b_exp - a_exp

            # Align a's mantissa
            if exp_diff >= MAN_BITS + 5:
                aligned_a.next = 0
                aligned_b.next = b_man_ext
            else:
                aligned_a.next = a_man_ext >> exp_diff
                aligned_b.next = b_man_ext

    @always_comb
    def add_mantissas():
        # Add/subtract based on signs
        if a_sign == b_sign:
            add_result.next = aligned_a + aligned_b
            final_sign.next = a_sign
        else:
            # Effective subtraction
            if aligned_a >= aligned_b:
                add_result.next = aligned_a - aligned_b
                final_sign.next = a_sign
            else:
                add_result.next = aligned_b - aligned_a
                final_sign.next = b_sign

    @always_comb
    def normalize_and_round():
        # Initialize with default values
        final_exp.next = larger_exp
        final_man.next = 0

        if add_result == 0:
            # Result is zero
            final_sign.next = 0  # +0 by convention
            final_exp.next = 0
            final_man.next = 0
        elif add_result[MAN_BITS + 5]:
            # Overflow in addition, shift right and increment exponent
            if larger_exp >= (1 << EXP_BITS) - 2:
                # Overflow to infinity
                final_exp.next = (1 << EXP_BITS) - 1
                final_man.next = 0
            else:
                final_exp.next = larger_exp + 1
                # Round
                guard = bool(add_result[2])
                round_bit = bool(add_result[1])
                sticky = bool(add_result[0])

                temp_man = (add_result >> 3) & ((1 << MAN_BITS) - 1)
                if round_bit and (sticky or guard):
                    if temp_man == (1 << MAN_BITS) - 1:
                        temp_man = 0
                        final_exp.next = final_exp + 1
                        if final_exp == (1 << EXP_BITS) - 1:
                            # Overflow to infinity
                            final_man.next = 0
                        else:
                            final_man.next = temp_man
                    else:
                        final_man.next = temp_man + 1
                else:
                    final_man.next = temp_man
        else:
            # Find position of leading 1
            lead_pos = MAN_BITS + 4
            while lead_pos >= 0 and not add_result[lead_pos]:
                lead_pos = lead_pos - 1

            if lead_pos < 0:
                # Result is zero (shouldn't happen if we checked earlier)
                final_exp.next = 0
                final_man.next = 0
            else:
                # Calculate normalization shift
                shift_left = (MAN_BITS + 4) - lead_pos

                if larger_exp <= shift_left:
                    # Result will be denormalized or zero
                    if larger_exp == 0:
                        # Already at minimum exponent
                        final_exp.next = 0
                        final_man.next = (add_result >> 2) & ((1 << MAN_BITS) - 1)
                    else:
                        # Shift as much as possible
                        final_exp.next = 0
                        shift_amount = larger_exp
                        shifted = add_result << shift_amount
                        final_man.next = (shifted >> 2) & ((1 << MAN_BITS) - 1)
                else:
                    # Normal case - can fully normalize
                    final_exp.next = larger_exp - shift_left
                    shifted = add_result << shift_left

                    # Round
                    guard = bool(shifted[2])
                    round_bit = bool(shifted[1])
                    sticky = bool(shifted[0])

                    temp_man = (shifted >> 2) & ((1 << MAN_BITS) - 1)
                    if round_bit and (sticky or guard):
                        if (
                            temp_man == (1 << MAN_BITS) - 1
                            and final_exp == (1 << EXP_BITS) - 2
                        ):
                            # Round up would overflow to infinity
                            final_exp.next = (1 << EXP_BITS) - 1
                            final_man.next = 0
                        elif temp_man == (1 << MAN_BITS) - 1:
                            # Round up carries to exponent
                            final_man.next = 0
                            final_exp.next = final_exp + 1
                        else:
                            final_man.next = temp_man + 1
                    else:
                        final_man.next = temp_man

    @always_comb
    def handle_special_cases():
        if a_is_nan or b_is_nan or (a_is_inf and b_is_inf and a_sign != b_sign):
            # NaN cases
            result.next = (
                (1 << (WIDTH - 1))
                | ((1 << EXP_BITS) - 1) << MAN_BITS
                | (1 << (MAN_BITS - 1))
            )
        elif a_is_inf:
            # Infinity + anything = infinity with a's sign
            result.next = (a_sign << (WIDTH - 1)) | ((1 << EXP_BITS) - 1) << MAN_BITS
        elif b_is_inf:
            # Anything + infinity = infinity with b's sign
            result.next = (b_sign << (WIDTH - 1)) | ((1 << EXP_BITS) - 1) << MAN_BITS
        elif a_is_zero and b_is_zero:
            # Special case for zero + zero
            if a_sign and b_sign:
                # -0 + -0 = -0
                result.next = 1 << (WIDTH - 1)
            else:
                # Other zero combinations = +0
                result.next = 0
        elif a_is_zero:
            # 0 + b = b
            result.next = input_b
        elif b_is_zero:
            # a + 0 = a
            result.next = input_a
        elif add_result == 0:
            # Result is exactly zero
            result.next = 0  # +0 by convention
        else:
            # Normal or denormal result
            result.next = (
                (final_sign << (WIDTH - 1)) | (final_exp << MAN_BITS) | final_man
            )

    # Register the output
    @always(clk.posedge)
    def reg_output():
        if rst:
            output_z.next = 0
        else:
            output_z.next = result

    return (
        extract_components,
        align_mantissas,
        add_mantissas,
        normalize_and_round,
        handle_special_cases,
        reg_output,
    )
