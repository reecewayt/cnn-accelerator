from myhdl import *
import math


def fp_mac(clk, reset, a, b, c, result, width=32):
    """
    Floating Point Multiply-Accumulate (MAC) Unit

    Parameters:
    -----------
    clk : Signal
        Clock signal
    reset : Signal
        Reset signal (active high)
    a : SignalVector
        First input operand (width bits)
    b : SignalVector
        Second input operand (width bits)
    c : SignalVector
        Accumulator input (width bits)
    result : SignalVector
        Output result (width bits)
    width : int
        Bit width of floating point format (16, 32, 64, or 128)

    Returns:
    --------
    myhdl instances
    """

    # Define parameters based on floating point format width
    if width == 16:  # Half precision
        EXP_BITS = 5
        MANT_BITS = 10
    elif width == 32:  # Single precision
        EXP_BITS = 8
        MANT_BITS = 23
    elif width == 64:  # Double precision
        EXP_BITS = 11
        MANT_BITS = 52
    elif width == 128:  # Quadruple precision
        EXP_BITS = 15
        MANT_BITS = 112
    else:
        raise ValueError("Unsupported floating point width")

    # Calculate bias for the exponent
    BIAS = (2 ** (EXP_BITS - 1)) - 1

    # Internal signals for unpacking floating point numbers
    a_sign = Signal(bool(0))
    a_exp = Signal(intbv(0)[EXP_BITS:])
    a_mant = Signal(intbv(0)[MANT_BITS:])

    b_sign = Signal(bool(0))
    b_exp = Signal(intbv(0)[EXP_BITS:])
    b_mant = Signal(intbv(0)[MANT_BITS:])

    c_sign = Signal(bool(0))
    c_exp = Signal(intbv(0)[EXP_BITS:])
    c_mant = Signal(intbv(0)[MANT_BITS:])

    # Internal signals for multiplication
    mul_sign = Signal(bool(0))
    mul_exp = Signal(intbv(0)[EXP_BITS + 1 :])  # Extra bit for potential overflow
    mul_mant = Signal(intbv(0)[2 * MANT_BITS + 2 :])  # Full precision multiplication

    # Internal signals for normalization and addition
    norm_mul_exp = Signal(intbv(0)[EXP_BITS + 1 :])
    norm_mul_mant = Signal(
        intbv(0)[MANT_BITS + 3 :]
    )  # +3 for hidden bit, guard, and round

    add_sign = Signal(bool(0))
    add_exp = Signal(intbv(0)[EXP_BITS + 2 :])  # Extra bits for alignment
    add_mant = Signal(
        intbv(0)[MANT_BITS + 5 :]
    )  # Extra bits for alignment and rounding

    # Pipeline registers
    stage1_valid = Signal(bool(0))
    stage2_valid = Signal(bool(0))

    # Status flags
    overflow = Signal(bool(0))
    underflow = Signal(bool(0))

    @always_seq(clk.posedge, reset=reset)
    def unpack():
        """First stage: Unpack floating point numbers"""
        # Extract sign, exponent, and mantissa from inputs
        a_sign.next = a[width - 1]
        a_exp.next = a[width - 2 : width - 2 - EXP_BITS]
        a_mant.next = a[width - 2 - EXP_BITS :]

        b_sign.next = b[width - 1]
        b_exp.next = b[width - 2 : width - 2 - EXP_BITS]
        b_mant.next = b[width - 2 - EXP_BITS :]

        c_sign.next = c[width - 1]
        c_exp.next = c[width - 2 : width - 2 - EXP_BITS]
        c_mant.next = c[width - 2 - EXP_BITS :]

        stage1_valid.next = True

    @always_seq(clk.posedge, reset=reset)
    def multiply():
        """Second stage: Multiplication"""
        if stage1_valid:
            # Compute sign of multiplication
            mul_sign.next = a_sign ^ b_sign

            # Check for special cases (0, infinity, NaN)
            if (a_exp == 0) or (b_exp == 0):
                # If either input is zero or denormal
                mul_exp.next = 0
                mul_mant.next = 0
            elif (a_exp == 2**EXP_BITS - 1) or (b_exp == 2**EXP_BITS - 1):
                # If either input is Inf or NaN
                mul_exp.next = 2**EXP_BITS - 1
                if ((a_exp == 2**EXP_BITS - 1) and (a_mant != 0)) or (
                    (b_exp == 2**EXP_BITS - 1) and (b_mant != 0)
                ):
                    # NaN propagation
                    mul_mant.next = 1
                else:
                    mul_mant.next = 0
            else:
                # Regular case: Multiply mantissas (with hidden bits) and add exponents
                # Add hidden bit '1' to mantissas
                a_full_mant = concat(intbv(1), a_mant)
                b_full_mant = concat(intbv(1), b_mant)

                # Multiply mantissas
                mul_mant.next = a_full_mant * b_full_mant

                # Add exponents and subtract bias
                mul_exp.next = a_exp + b_exp - BIAS

            # Normalize product
            if mul_mant[2 * MANT_BITS + 1]:
                # Product has 2.xxx format, shift right and increment exponent
                norm_mul_mant.next = mul_mant[2 * MANT_BITS + 1 : MANT_BITS - 1]
                norm_mul_exp.next = mul_exp + 1
            else:
                # Product has 1.xxx format, use as is
                norm_mul_mant.next = mul_mant[2 * MANT_BITS : MANT_BITS - 2]
                norm_mul_exp.next = mul_exp

            stage2_valid.next = True

    @always_seq(clk.posedge, reset=reset)
    def accumulate():
        """Third stage: Accumulate (add)"""
        if stage2_valid:
            # Align exponents for addition
            exp_diff = 0
            aligned_c_mant = 0

            # Special case handling
            c_is_special = (c_exp == 0) or (c_exp == 2**EXP_BITS - 1)
            mul_is_special = (norm_mul_exp == 0) or (norm_mul_exp >= 2**EXP_BITS - 1)

            if c_is_special and not mul_is_special:
                # C is special (0, Inf, NaN), mul is normal
                add_sign.next = c_sign
                add_exp.next = c_exp
                add_mant.next = c_mant
            elif mul_is_special and not c_is_special:
                # Mul is special, C is normal
                add_sign.next = mul_sign
                add_exp.next = norm_mul_exp
                add_mant.next = norm_mul_mant
            elif mul_is_special and c_is_special:
                # Both are special
                if (
                    (c_exp == 2**EXP_BITS - 1)
                    and (c_mant != 0)
                    or (norm_mul_exp >= 2**EXP_BITS - 1)
                    and (norm_mul_mant[MANT_BITS] != 0)
                ):
                    # NaN has priority
                    add_sign.next = 0
                    add_exp.next = 2**EXP_BITS - 1
                    add_mant.next = 1  # NaN
                else:
                    # Handle infinities
                    if c_exp == 2**EXP_BITS - 1 and norm_mul_exp >= 2**EXP_BITS - 1:
                        if c_sign == mul_sign:
                            # Same sign infinities add to infinity
                            add_sign.next = c_sign
                            add_exp.next = 2**EXP_BITS - 1
                            add_mant.next = 0
                        else:
                            # Different sign infinities result in NaN
                            add_sign.next = 0
                            add_exp.next = 2**EXP_BITS - 1
                            add_mant.next = 1
                    elif c_exp == 2**EXP_BITS - 1:
                        # C is infinity
                        add_sign.next = c_sign
                        add_exp.next = 2**EXP_BITS - 1
                        add_mant.next = 0
                    else:
                        # Mul is infinity
                        add_sign.next = mul_sign
                        add_exp.next = 2**EXP_BITS - 1
                        add_mant.next = 0
            else:
                # Normal case: Both are normal numbers, need to align and add
                if norm_mul_exp > c_exp:
                    # Mul has larger exponent, shift c
                    add_exp.next = norm_mul_exp
                    exp_diff = norm_mul_exp - c_exp

                    # Add hidden bit to c_mant
                    c_full_mant = concat(intbv(1), c_mant, intbv(0)[3:])

                    # Shift c mantissa right by exp_diff (align)
                    if exp_diff > MANT_BITS + 3:
                        aligned_c_mant = 0  # Too small to matter
                    else:
                        aligned_c_mant = c_full_mant >> exp_diff

                    # Add or subtract mantissas based on signs
                    if mul_sign == c_sign:
                        # Same sign: add
                        add_sign.next = mul_sign
                        add_mant.next = norm_mul_mant + aligned_c_mant
                    else:
                        # Different signs: subtract
                        if norm_mul_mant >= aligned_c_mant:
                            add_sign.next = mul_sign
                            add_mant.next = norm_mul_mant - aligned_c_mant
                        else:
                            add_sign.next = c_sign
                            add_mant.next = aligned_c_mant - norm_mul_mant
                elif c_exp > norm_mul_exp:
                    # C has larger exponent, shift mul
                    add_exp.next = c_exp
                    exp_diff = c_exp - norm_mul_exp

                    # Shift mul mantissa right by exp_diff (align)
                    if exp_diff > MANT_BITS + 3:
                        aligned_mul_mant = 0  # Too small to matter
                    else:
                        aligned_mul_mant = norm_mul_mant >> exp_diff

                    # Add or subtract mantissas based on signs
                    if mul_sign == c_sign:
                        # Same sign: add
                        add_sign.next = c_sign
                        add_mant.next = c_mant + aligned_mul_mant
                    else:
                        # Different signs: subtract
                        if c_mant >= aligned_mul_mant:
                            add_sign.next = c_sign
                            add_mant.next = c_mant - aligned_mul_mant
                        else:
                            add_sign.next = mul_sign
                            add_mant.next = aligned_mul_mant - c_mant
                else:
                    # Equal exponents, no shifting needed
                    add_exp.next = c_exp

                    # Add or subtract mantissas based on signs
                    if mul_sign == c_sign:
                        # Same sign: add
                        add_sign.next = c_sign
                        add_mant.next = norm_mul_mant + c_mant
                    else:
                        # Different signs: subtract
                        if norm_mul_mant >= c_mant:
                            add_sign.next = mul_sign
                            add_mant.next = norm_mul_mant - c_mant
                        else:
                            add_sign.next = c_sign
                            add_mant.next = c_mant - norm_mul_mant

            # Normalize result
            if add_mant[MANT_BITS + 4]:
                # Result has overflowed, shift right and increment exponent
                result.next = concat(add_sign, add_exp + 1, add_mant[MANT_BITS + 3 : 3])
                if add_exp + 1 >= 2**EXP_BITS - 1:
                    overflow.next = True
            elif add_mant[MANT_BITS + 3]:
                # Result is normalized
                result.next = concat(add_sign, add_exp, add_mant[MANT_BITS + 2 : 2])
            else:
                # Result needs normalization (leading zeros)
                # Count leading zeros and adjust
                leading_zeros = 0
                temp_mant = add_mant
                while temp_mant[MANT_BITS + 2] == 0 and leading_zeros < add_exp:
                    leading_zeros += 1
                    temp_mant = temp_mant << 1

                if leading_zeros >= add_exp:
                    # Result is denormalized
                    result.next = concat(
                        add_sign,
                        intbv(0)[EXP_BITS:],
                        add_mant[MANT_BITS:0] << (add_exp - 1),
                    )
                    if add_mant != 0:
                        underflow.next = True
                else:
                    # Result is normalized with adjusted exponent
                    result.next = concat(
                        add_sign, add_exp - leading_zeros, temp_mant[MANT_BITS + 2 : 2]
                    )

    return instances()
