from myhdl import *
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.utils.fp_defs import E4M3Format


@block
def fp8_e4m3_multiply(input_a, input_b, output_z, start, done, clk, rst):
    """
    State machine-based E4M3 floating-point multiplier
    Parameters:
    - input_a, input_b: Input E4M3 operands (8-bit each)
    - output_z: Output E4M3 product (8-bit)
    - start: Control signal to start computation (active high)
    - done: Signal indicating computation is complete (active high)
    - clk, rst: Clock and reset signals
    """
    # Constants from E4M3Format
    WIDTH = E4M3Format.WIDTH  # 8
    EXP_BITS = E4M3Format.EXP_BITS  # 4
    MAN_BITS = E4M3Format.MAN_BITS  # 3
    EXP_BIAS = E4M3Format.EXP_BIAS  # 7

    # State definitions
    t_State = enum(
        "IDLE",
        "UNPACK",
        "SPECIAL_CASES",
        "MULTIPLY",
        "NORMALIZE",
        "ROUND",
        "PACK",
        "PUT_Z",
    )
    state = Signal(t_State.IDLE)

    # Internal registers for inputs and output
    a = Signal(intbv(0)[WIDTH:])
    b = Signal(intbv(0)[WIDTH:])
    z = Signal(intbv(0)[WIDTH:])

    # Unpacked fields
    a_sign = Signal(bool(0))
    b_sign = Signal(bool(0))
    z_sign = Signal(bool(0))

    # Use similar ranges as in the adder for exponents
    a_exp = Signal(intbv(0, min=-(2 ** (EXP_BITS)), max=2 ** (EXP_BITS)))
    b_exp = Signal(intbv(0, min=-(2 ** (EXP_BITS)), max=2 ** (EXP_BITS)))
    z_exp = Signal(intbv(0, min=-(2 ** (EXP_BITS)), max=2 ** (EXP_BITS)))

    # Extended mantissa to handle implicit bit
    a_man = Signal(intbv(0)[MAN_BITS + 1 :])
    b_man = Signal(intbv(0)[MAN_BITS + 1 :])

    # Product needs twice the width
    product = Signal(intbv(0)[2 * (MAN_BITS + 1) :])

    # Final mantissa with extra bits for rounding
    z_man = Signal(intbv(0)[MAN_BITS + 3 :])

    # Special case flags
    a_is_zero = Signal(bool(0))
    b_is_zero = Signal(bool(0))
    a_is_nan = Signal(bool(0))
    b_is_nan = Signal(bool(0))

    # Rounding bits
    guard = Signal(bool(0))
    round_bit = Signal(bool(0))
    sticky = Signal(bool(0))

    # Output signals
    s_output_z = Signal(intbv(0)[WIDTH:])
    s_done = Signal(bool(0))

    @always_seq(clk.posedge, reset=rst)
    def state_machine():
        if rst:
            state.next = t_State.IDLE
            s_done.next = 0
        else:
            if state == t_State.IDLE:
                s_done.next = 0
                if start:
                    a.next = input_a
                    b.next = input_b
                    state.next = t_State.UNPACK

            elif state == t_State.UNPACK:
                # Extract components from operands
                a_sign.next = bool(a[WIDTH - 1])
                a_exp.next = a[WIDTH - 1 : MAN_BITS] - EXP_BIAS
                b_sign.next = bool(b[WIDTH - 1])
                b_exp.next = b[WIDTH - 1 : MAN_BITS] - EXP_BIAS

                # Handle normal/denormal numbers
                if a[WIDTH - 1 : MAN_BITS] != 0:  # Normal number
                    a_man.next = concat(intbv(1)[1:], a[MAN_BITS:])
                else:  # Denormal number
                    a_man.next = concat(intbv(0)[1:], a[MAN_BITS:])
                    a_exp.next = -EXP_BIAS + 1

                if b[WIDTH - 1 : MAN_BITS] != 0:  # Normal number
                    b_man.next = concat(intbv(1)[1:], b[MAN_BITS:])
                else:  # Denormal number
                    b_man.next = concat(intbv(0)[1:], b[MAN_BITS:])
                    b_exp.next = -EXP_BIAS + 1

                # Detect special cases
                a_is_zero.next = (a[WIDTH - 1 : MAN_BITS] == 0) and (a[MAN_BITS:] == 0)
                b_is_zero.next = (b[WIDTH - 1 : MAN_BITS] == 0) and (b[MAN_BITS:] == 0)

                # NaN detection (all 1s in exponent + non-zero mantissa)
                a_is_nan.next = (a[WIDTH - 1 : MAN_BITS] == (1 << EXP_BITS) - 1) and (
                    a[MAN_BITS:] != 0
                )
                b_is_nan.next = (b[WIDTH - 1 : MAN_BITS] == (1 << EXP_BITS) - 1) and (
                    b[MAN_BITS:] != 0
                )

                state.next = t_State.SPECIAL_CASES

            elif state == t_State.SPECIAL_CASES:
                # Set sign for the result (XOR of input signs)
                z_sign.next = a_sign ^ b_sign

                # Check for NaN cases
                if a_is_nan or b_is_nan:
                    # NaN result
                    z.next = ((1 << EXP_BITS) - 1) << MAN_BITS | (1 << (MAN_BITS - 1))
                    state.next = t_State.PUT_Z

                # Check for zero cases
                elif a_is_zero or b_is_zero:
                    # Zero result with correct sign
                    z.next = z_sign << (WIDTH - 1)
                    state.next = t_State.PUT_Z

                else:
                    # Regular multiplication
                    state.next = t_State.MULTIPLY

            elif state == t_State.MULTIPLY:
                # Add exponents (both are already bias adjusted)
                z_exp.next = a_exp + b_exp

                # Multiply mantissas
                product.next = a_man * b_man

                state.next = t_State.NORMALIZE

            elif state == t_State.NORMALIZE:
                # For product = 0x60 (binary 01100000), we want to extract:
                # Bit position: [7][6][5][4][3][2][1][0]
                # Value:         0  1  1  0  0  0  0  0

                # Check if bit 7 is set (implicit overflow)
                if product[2 * (MAN_BITS + 1) - 1]:  # Bit 7
                    # Need to shift right and adjust exponent
                    # Extract bits [6:4] as mantissa - in MyHDL slice [7:4]
                    z_man.next = product[
                        2 * (MAN_BITS + 1) - 1 : 2 * (MAN_BITS + 1) - MAN_BITS - 1
                    ]
                    z_exp.next = z_exp + 1
                else:
                    # No overflow, normalized product
                    # Extract bits [5:3] as mantissa - in MyHDL slice [6:3]
                    z_man.next = product[
                        2 * (MAN_BITS + 1) - 2 : 2 * (MAN_BITS + 1) - MAN_BITS - 2
                    ]

                # Setup rounding bits from bits [2:0]
                guard.next = bool(product[2])
                round_bit.next = bool(product[1])
                sticky.next = bool(product[0])

                state.next = t_State.ROUND

            elif state == t_State.ROUND:
                # For E4M3, our mantissa is now in z_man[MAN_BITS:0]
                # Round to nearest even
                if guard and (round_bit or sticky or z_man[0]):
                    # Add 1 to the LSB of our mantissa
                    new_man = z_man + 1
                    z_man.next = new_man

                    # Check if rounding caused overflow
                    if new_man[MAN_BITS + 1]:  # If overflow in mantissa
                        # Need to right shift and adjust exponent
                        z_man.next = new_man >> 1
                        z_exp.next = z_exp + 1

                state.next = t_State.PACK

            elif state == t_State.PACK:
                # For E4M3, we need the 3-bit mantissa

                # Check for underflow/overflow
                if z_exp < -EXP_BIAS:
                    # Underflow to zero
                    z.next = z_sign << (WIDTH - 1)
                elif z_exp >= EXP_BIAS:
                    # Overflow to max value
                    z.next = (
                        (z_sign << (WIDTH - 1))
                        | ((1 << EXP_BITS) - 1) << MAN_BITS
                        | ((1 << MAN_BITS) - 2)
                    )
                else:
                    # Normal case
                    # Extract our 3-bit mantissa for the final result - bits [2:0]
                    final_man = z_man[MAN_BITS:0]

                    # Assemble the final 8-bit E4M3 value
                    biased_exp = z_exp + EXP_BIAS
                    z.next = (
                        (z_sign << (WIDTH - 1)) | (biased_exp << MAN_BITS) | final_man
                    )

                state.next = t_State.PUT_Z

            elif state == t_State.PUT_Z:
                s_output_z.next = z
                s_done.next = 1
                state.next = t_State.IDLE

    @always_comb
    def output_logic():
        # Connect internal signals to outputs
        output_z.next = s_output_z
        done.next = s_done

    return instances()
