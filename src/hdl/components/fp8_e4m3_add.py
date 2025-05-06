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

    # Rounding bits
    guard = Signal(bool(0))
    round_bit = Signal(bool(0))
    sticky = Signal(bool(0))

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

                state.next = t_State.SPECIAL_CASES

            elif state == t_State.SPECIAL_CASES:
                # Check for NaN (in E4M3: exp=max and mantissa!=0)
                if (
                    a[WIDTH - 1 : MAN_BITS] == (1 << EXP_BITS) - 1 and a[MAN_BITS:] != 0
                ) or (
                    b[WIDTH - 1 : MAN_BITS] == (1 << EXP_BITS) - 1 and b[MAN_BITS:] != 0
                ):
                    # Use max negative value for NaN in E4M3
                    z.next = (
                        (1 << (WIDTH - 1))
                        | ((1 << EXP_BITS) - 1) << MAN_BITS
                        | ((1 << MAN_BITS) - 1)
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

                else:
                    state.next = t_State.ALIGN

            elif state == t_State.ALIGN:
                # This step will repeatedly shift the smaller exponent until they are equal
                if a_e > b_e:
                    b_e.next = b_e + 1
                    # Shift with sticky bit
                    b_m.next = b_m >> 1
                    if b_m[0]:  # Save shifted-out bit for better rounding
                        b_m.next[0] = 1

                elif a_e < b_e:
                    a_e.next = a_e + 1
                    # Shift with sticky bit
                    a_m.next = a_m >> 1
                    if a_m[0]:  # Save shifted-out bit for better rounding
                        a_m.next[0] = 1

                else:
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

                # Handle overflow - clamp to max value
                if z_e >= EXP_BIAS:
                    z.next[WIDTH - 1 : MAN_BITS] = (1 << EXP_BITS) - 1
                    z.next[MAN_BITS:] = (1 << MAN_BITS) - 1
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
