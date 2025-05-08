from myhdl import *
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.utils.fp_defs import E4M3Format


@block
def fp8_e4m3_add(input_a, input_b, output_z, start, done, clk, rst):
    """
    E4M3 floating-point adder (implemented with state machine)
    Parameters:
    - input_a, input_b: Input E4M3 operands (8-bit each)
    - output_z: Output E4M3 sum (8-bit)
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
        "ALIGN",
        "ADD_0",
        "ADD_1",
        "NORMALISE_1",
        "NORMALISE_2",
        "ROUND",
        "PACK",
        "PUT_Z",
    )
    state = Signal(t_State.IDLE)

    # Internal registers
    a = Signal(intbv(0)[WIDTH:])
    b = Signal(intbv(0)[WIDTH:])
    z = Signal(intbv(0)[WIDTH:])

    # Unpacked fields
    a_s = Signal(bool(0))  # sign bit
    b_s = Signal(bool(0))
    z_s = Signal(bool(0))

    # Change these lines
    a_e = Signal(intbv(0, min=-(2 ** (EXP_BITS)), max=2 ** (EXP_BITS)))
    b_e = Signal(intbv(0, min=-(2 ** (EXP_BITS)), max=2 ** (EXP_BITS)))
    z_e = Signal(intbv(0, min=-(2 ** (EXP_BITS)), max=2 ** (EXP_BITS)))

    a_m = Signal(intbv(0)[MAN_BITS + 2 :])  # mantissa (+2 bits for guard/round)
    b_m = Signal(intbv(0)[MAN_BITS + 2 :])
    z_m = Signal(intbv(0)[MAN_BITS + 1 :])

    # Rounding bits:
    # These three bits are used to implement IEEE 754-style rounding for floating-point operations
    #
    # guard: The first bit that's shifted out of the mantissa during alignment or normalization.
    #        It's the bit immediately to the right of the least significant bit (LSB) of the final result.
    #        The guard bit provides one extra bit of precision for intermediate calculations.
    #
    # round_bit: The second bit that's shifted out of the mantissa. It's the bit immediately to the
    #            right of the guard bit. This bit helps determine the direction of rounding.
    #
    # sticky: Logical OR of all remaining bits that are shifted out beyond the round bit.
    #         Once set to 1, it stays 1 (hence "sticky"). It indicates if any 1 bits exist
    #         beyond the round bit position, which affects the rounding decision.
    #
    # Together, these bits implement "round to nearest, ties to even" rounding:
    # - If guard=0: Round down (truncate)
    # - If guard=1 and (round=1 or sticky=1 or LSB=1): Round up
    # - If guard=1 and round=0 and sticky=0 and LSB=0: Round down (to even)
    guard = Signal(bool(0))
    round_bit = Signal(bool(0))
    sticky = Signal(bool(0))

    # Exponent signals for handling special cases
    # Add this with your other signal definitions (before the state machine)
    exp_diff = Signal(intbv(0, min=-(2 ** (EXP_BITS + 1)), max=2 ** (EXP_BITS + 1)))
    max_shifts = Signal(intbv(MAN_BITS + 2)[4:])  # 5 for E4M3 format, using 4 bits

    # Addition result
    sum_val = Signal(intbv(0)[MAN_BITS + 3 :])  # Extra bit for potential overflow

    # Output register
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
                # Extract components
                a_s.next = bool(a[WIDTH - 1])
                a_e.next = a[WIDTH - 1 : MAN_BITS] - EXP_BIAS
                a_m.next = intbv(0)[MAN_BITS + 2 :]

                b_s.next = bool(b[WIDTH - 1])
                b_e.next = b[WIDTH - 1 : MAN_BITS] - EXP_BIAS
                b_m.next = intbv(0)[MAN_BITS + 2 :]

                # Handle normal numbers with implicit bit
                if a[WIDTH - 1 : MAN_BITS] != 0:  # If exponent not zero
                    a_m.next = concat(intbv(1)[1:], a[MAN_BITS:], intbv(0)[1:])
                else:
                    # Denormal handling
                    a_m.next = concat(intbv(0)[1:], a[MAN_BITS:], intbv(0)[1:])
                    a_e.next = -EXP_BIAS + 1

                if b[WIDTH - 1 : MAN_BITS] != 0:
                    b_m.next = concat(intbv(1)[1:], b[MAN_BITS:], intbv(0)[1:])
                else:
                    # Denormal handling
                    b_m.next = concat(intbv(0)[1:], b[MAN_BITS:], intbv(0)[1:])
                    b_e.next = -EXP_BIAS + 1

                # find exponent difference
                if (
                    a[WIDTH - 1 : MAN_BITS] - EXP_BIAS
                    > b[WIDTH - 1 : MAN_BITS] - EXP_BIAS
                ):
                    exp_diff.next = (a[WIDTH - 1 : MAN_BITS] - EXP_BIAS) - (
                        b[WIDTH - 1 : MAN_BITS] - EXP_BIAS
                    )
                else:
                    exp_diff.next = (b[WIDTH - 1 : MAN_BITS] - EXP_BIAS) - (
                        a[WIDTH - 1 : MAN_BITS] - EXP_BIAS
                    )

                state.next = t_State.SPECIAL_CASES

            elif state == t_State.SPECIAL_CASES:
                # Check for NaN (in E4M3: exp=1111 and mantissa=111)
                if (
                    a[WIDTH - 1 : MAN_BITS] == (1 << EXP_BITS) - 1
                    and a[MAN_BITS:] == (1 << MAN_BITS) - 1
                ) or (
                    b[WIDTH - 1 : MAN_BITS] == (1 << EXP_BITS) - 1
                    and b[MAN_BITS:] == (1 << MAN_BITS) - 1
                ):
                    # NaN in E4M3 is represented as sign bit (0 for +NaN) with exponent=1111 and mantissa=111
                    z.next = (
                        (z_s << (WIDTH - 1))  # Sign bit
                        | ((1 << EXP_BITS) - 1) << MAN_BITS  # Exponent 1111
                        | ((1 << MAN_BITS) - 1)  # Mantissa 111
                    )
                    state.next = t_State.PUT_Z

                # If a is zero, return b
                elif a[WIDTH - 1 : MAN_BITS] == 0 and a[MAN_BITS:] == 0:
                    if b[WIDTH - 1 : MAN_BITS] == 0 and b[MAN_BITS:] == 0:
                        # Both zeros - return signed zero (negative if both negative)
                        z.next = (a_s & b_s) << (WIDTH - 1)
                    else:
                        z.next = b
                    state.next = t_State.PUT_Z

                # If b is zero, return a
                elif b[WIDTH - 1 : MAN_BITS] == 0 and b[MAN_BITS:] == 0:
                    z.next = a
                    state.next = t_State.PUT_Z

                # Check for operations with max values that would overflow
                elif (
                    (
                        a[WIDTH - 1 : MAN_BITS] == (1 << EXP_BITS) - 1
                        and a[MAN_BITS:] == (1 << MAN_BITS) - 2
                    )  # if a is max (0x7E)
                    or (
                        b[WIDTH - 1 : MAN_BITS] == (1 << EXP_BITS) - 1
                        and b[MAN_BITS:] == (1 << MAN_BITS) - 2
                    )
                ) and (
                    a_s == b_s
                ):  # Only if same sign (addition would overflow)
                    # Set result to max value
                    z.next = (
                        ((a_s & b_s) << (WIDTH - 1))
                        | ((1 << EXP_BITS) - 1) << MAN_BITS
                        | ((1 << MAN_BITS) - 2)
                    )
                    state.next = t_State.PUT_Z

                else:
                    state.next = t_State.ALIGN

            elif state == t_State.ALIGN:
                # This step will repeatedly shift the smaller exponent until they are equal
                if a_e > b_e:
                    if exp_diff > max_shifts:
                        # For very large differences, the smaller operand is effectively zero
                        # Skip computation and just use the larger operand (a)
                        z.next = a
                        state.next = t_State.PUT_Z
                    else:
                        # Regular alignment - shift b
                        b_e.next = b_e + 1
                        # Shift with sticky bit
                        b_m.next = b_m >> 1
                        if b_m[0]:  # Save shifted-out bit for better rounding
                            b_m.next[0] = 1
                elif a_e < b_e:

                    if exp_diff > max_shifts:
                        # For very large differences, the smaller operand is effectively zero
                        # Skip computation and just use the larger operand (b)
                        z.next = b
                        state.next = t_State.PUT_Z
                    else:
                        # Regular alignment - shift a
                        a_e.next = a_e + 1
                        # Shift with sticky bit
                        a_m.next = a_m >> 1
                        if a_m[0]:  # Save shifted-out bit for better rounding
                            a_m.next[0] = 1
                else:
                    # Exponents are equal, proceed to addition
                    state.next = t_State.ADD_0

            elif state == t_State.ADD_0:
                z_e.next = a_e

                if a_s == b_s:
                    # Same sign - add mantissas
                    sum_val.next = a_m + b_m
                    z_s.next = a_s
                else:
                    # Different signs - subtract the smaller from the larger
                    if a_m >= b_m:
                        sum_val.next = a_m - b_m
                        z_s.next = a_s
                    else:
                        sum_val.next = b_m - a_m
                        z_s.next = b_s

                state.next = t_State.ADD_1

            elif state == t_State.ADD_1:
                if sum_val[MAN_BITS + 2]:  # If overflow bit is set
                    z_m.next = sum_val[MAN_BITS + 3 : 2]
                    guard.next = bool(sum_val[1])
                    round_bit.next = bool(sum_val[0])
                    sticky.next = False
                    z_e.next = z_e + 1
                else:
                    z_m.next = sum_val[MAN_BITS + 2 : 1]
                    guard.next = bool(sum_val[0])
                    round_bit.next = False
                    sticky.next = False

                state.next = t_State.NORMALISE_1

            elif state == t_State.NORMALISE_1:
                # Left normalization (for subnormal results)
                if z_m[MAN_BITS] == 0 and z_e > -EXP_BIAS + 1:
                    z_e.next = z_e - 1
                    z_m.next = z_m << 1
                    z_m.next[0] = guard
                    guard.next = round_bit
                    round_bit.next = False
                else:
                    state.next = t_State.NORMALISE_2

            elif state == t_State.NORMALISE_2:
                # Right normalization (for potential underflow)
                if z_e < -EXP_BIAS + 1:
                    z_e.next = z_e + 1
                    # Shift with sticky bit update
                    guard.next = z_m[0]
                    z_m.next = z_m >> 1
                    round_bit.next = guard
                    sticky.next = sticky | round_bit
                else:
                    state.next = t_State.ROUND

            elif state == t_State.ROUND:
                # Round to nearest even
                if guard and (round_bit or sticky or z_m[0]):
                    z_m.next = z_m + 1
                    if z_m == (1 << MAN_BITS) - 1:
                        z_e.next = z_e + 1

                state.next = t_State.PACK

            elif state == t_State.PACK:
                # Default packing
                z.next[MAN_BITS:] = z_m[MAN_BITS:0]
                z.next[WIDTH - 1 : MAN_BITS] = z_e + EXP_BIAS
                z.next[WIDTH - 1] = z_s

                # Handle denormal results
                if z_e == -EXP_BIAS + 1 and z_m[MAN_BITS] == 0:
                    z.next[WIDTH - 1 : MAN_BITS] = 0

                # Fix sign for zero result
                if z_e <= -EXP_BIAS + 1 and z_m == 0:
                    z.next[WIDTH - 1] = 0  # +0 for zero result

                # Handle overflow - clamp to max value but avoid NaN
                if z_e >= EXP_BIAS:
                    # Set to maximum representable value without causing NaN
                    # Maximum value has exponent 1111 and mantissa 110
                    z.next[WIDTH - 1 : MAN_BITS] = (
                        1 << EXP_BITS
                    ) - 1  # Exponent = 1111
                    z.next[MAN_BITS:] = (1 << MAN_BITS) - 2  # Mantissa = 110 (not 111)
                    # Keep the sign bit

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
