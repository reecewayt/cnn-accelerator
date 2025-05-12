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

    # Use a wider range for exponents to handle intermediate calculations
    a_exp = Signal(intbv(0, min=-(2 ** (EXP_BITS + 2)), max=2 ** (EXP_BITS + 2)))
    b_exp = Signal(intbv(0, min=-(2 ** (EXP_BITS + 2)), max=2 ** (EXP_BITS + 2)))
    z_exp = Signal(intbv(0, min=-(2 ** (EXP_BITS + 2)), max=2 ** (EXP_BITS + 2)))

    # Extended mantissa to handle implicit bit
    a_man = Signal(intbv(0)[MAN_BITS + 1 :])
    b_man = Signal(intbv(0)[MAN_BITS + 1 :])

    final_man = Signal(intbv(0)[MAN_BITS:])
    shift_amount = Signal(intbv(0)[EXP_BITS + 1 :])
    denorm_man = Signal(intbv(0)[MAN_BITS + 3 :])
    biased_exp = Signal(intbv(0)[EXP_BITS:])

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
                    a_exp.next = 1 - EXP_BIAS

                if b[WIDTH - 1 : MAN_BITS] != 0:  # Normal number
                    b_man.next = concat(intbv(1)[1:], b[MAN_BITS:])
                else:  # Denormal number
                    b_man.next = concat(intbv(0)[1:], b[MAN_BITS:])
                    b_exp.next = 1 - EXP_BIAS

                # Detect special cases
                a_is_zero.next = (a[WIDTH - 1 : MAN_BITS] == 0) and (a[MAN_BITS:] == 0)
                b_is_zero.next = (b[WIDTH - 1 : MAN_BITS] == 0) and (b[MAN_BITS:] == 0)

                # NaN detection (all 1s in exponent + all 1s in mantissa)
                a_is_nan.next = (a[WIDTH - 1 : MAN_BITS] == (1 << EXP_BITS) - 1) and (
                    a[MAN_BITS:] == (1 << MAN_BITS) - 1
                )
                b_is_nan.next = (b[WIDTH - 1 : MAN_BITS] == (1 << EXP_BITS) - 1) and (
                    b[MAN_BITS:] == (1 << MAN_BITS) - 1
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
                # z_exp.next = a_exp + b_exp

                # Multiply mantissas but first handle denorm numbers
                if a_man[MAN_BITS] == 0 and a_man != 0:
                    # Denormalized number -> Shift left, and decrement exponent
                    a_man.next = a_man << 1
                    a_exp.next = a_exp - 1
                elif b_man[MAN_BITS] == 0 and b_man != 0:
                    # Denormalized number -> Shift left, and decrement exponent
                    b_man.next = b_man << 1
                    b_exp.next = b_exp - 1
                else:
                    # Normalized numbers
                    z_exp.next = a_exp + b_exp
                    product.next = a_man * b_man
                    if (a_exp + b_exp) >= EXP_BIAS + 2:
                        z.next = (z_sign << (WIDTH - 1)) | ((1 << EXP_BIAS) - 2)
                        state.next = t_State.PUT_Z

                    else:
                        state.next = t_State.NORMALIZE

            elif state == t_State.NORMALIZE:
                # Check if bit 7 is set (implicit overflow)
                if product[2 * (MAN_BITS + 1) - 1]:  # Bit 7
                    # Need to shift right and adjust exponent
                    z_man.next = product[
                        2 * (MAN_BITS + 1) - 1 : 2 * (MAN_BITS + 1) - MAN_BITS - 1
                    ]
                    z_exp.next = z_exp + 1
                else:
                    # No overflow, normalized product
                    z_man.next = product[
                        2 * (MAN_BITS + 1) - 2 : 2 * (MAN_BITS + 1) - MAN_BITS - 2
                    ]

                # Setup rounding bits from bits [2:0]
                guard.next = bool(product[2])
                round_bit.next = bool(product[1])
                sticky.next = bool(product[0])

                state.next = t_State.ROUND

            elif state == t_State.ROUND:
                # Round to nearest even
                if guard and (round_bit or sticky or z_man[0]):
                    # Add 1 to the LSB of our mantissa
                    z_man.next = z_man + 1

                    # Check if rounding caused overflow
                    if z_man == ((1 << (MAN_BITS + 1)) - 1):  # If overflow in mantissa
                        # Need to right shift and adjust exponent
                        z_exp.next = z_exp + 1

                state.next = t_State.PACK

            elif state == t_State.PACK:
                # Handle special cases first - check for complete underflow/overflow
                if z_exp < -EXP_BIAS - MAN_BITS:
                    # Complete underflow to zero
                    z.next = z_sign << (WIDTH - 1)
                elif z_exp < -EXP_BIAS:
                    # Gradual underflow - denormalized result
                    # Inline the shift amount calculation directly in the comparison
                    if (-EXP_BIAS - z_exp) <= MAN_BITS:
                        # Shift mantissa right and set exponent to 0
                        z.next = (z_sign << (WIDTH - 1)) | (
                            (z_man >> (-EXP_BIAS - z_exp))[MAN_BITS:0]
                        )
                    else:
                        # Too much underflow - flush to zero
                        z.next = z_sign << (WIDTH - 1)
                elif z_exp >= EXP_BIAS + 2:
                    # Overflow to max representable value (not NaN)
                    z.next = (
                        (z_sign << (WIDTH - 1))
                        | (((1 << EXP_BITS) - 1) << MAN_BITS)
                        | ((1 << MAN_BITS) - 2)
                    )
                else:
                    # Normal case - inline the biased exponent calculation
                    z.next = (
                        (z_sign << (WIDTH - 1))
                        | ((z_exp + EXP_BIAS) << MAN_BITS)
                        | z_man[MAN_BITS:0]
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
