@block
def fp8_mac(clk, reset, a, b, enable, result, overflow):
    """
    An E4M3 floating-point multiply-accumulate (MAC) unit.

    Parameters:
    - clk: Clock signal
    - reset: Active-high reset signal (clears accumulator)
    - a, b: Input E4M3 operands (8-bit each)
    - enable: Enable signal for accumulation
    - result: Output E4M3 result (accumulated value)
    - overflow: Overflow flag
    """
    # Intermediate signals
    product = Signal(intbv(0)[FP8_WIDTH:])
    acc = Signal(intbv(0)[FP8_WIDTH:])  # Internal accumulator

    # For addition, we'll need to decompose again
    acc_sign = Signal(bool(0))
    acc_exp = Signal(intbv(0)[EXP_BITS:])
    acc_man = Signal(intbv(0)[MAN_BITS:])

    prod_sign = Signal(bool(0))
    prod_exp = Signal(intbv(0)[EXP_BITS:])
    prod_man = Signal(intbv(0)[MAN_BITS:])

    # Extended mantissas for alignment
    acc_man_ext = Signal(intbv(0)[2 * MAN_BITS + EXP_BITS :])  # Extra bits for shifting
    prod_man_ext = Signal(intbv(0)[2 * MAN_BITS + EXP_BITS :])

    # Aligned mantissas
    aligned_acc = Signal(intbv(0)[2 * MAN_BITS + EXP_BITS + 1 :])  # +1 for sign
    aligned_prod = Signal(intbv(0)[2 * MAN_BITS + EXP_BITS + 1 :])

    # Sum result
    sum_val = Signal(intbv(0)[2 * MAN_BITS + EXP_BITS + 2 :])  # +1 for potential carry

    # Final normalized result components
    final_sign = Signal(bool(0))
    final_exp = Signal(intbv(0)[EXP_BITS:])
    final_man = Signal(intbv(0)[MAN_BITS:])

    # Special case flags
    acc_is_zero = Signal(bool(0))
    prod_is_zero = Signal(bool(0))
    acc_is_nan = Signal(bool(0))
    prod_is_nan = Signal(bool(0))
    result_is_nan = Signal(bool(0))

    # Instantiate the multiplier
    fp8_mult = fp8_multiply(a, b, product)

    @always_comb
    def decompose_acc_product():
        # Extract components from accumulator
        acc_sign.next = acc[FP8_WIDTH - 1 :] == 1
        acc_exp.next = acc[FP8_WIDTH - 1 : FP8_WIDTH - 1 - EXP_BITS]
        acc_man.next = acc[FP8_WIDTH - 1 - EXP_BITS :]

        # Extract components from product
        prod_sign.next = product[FP8_WIDTH - 1 :] == 1
        prod_exp.next = product[FP8_WIDTH - 1 : FP8_WIDTH - 1 - EXP_BITS]
        prod_man.next = product[FP8_WIDTH - 1 - EXP_BITS :]

        # Check for special cases
        acc_is_zero.next = (acc_exp == 0) and (acc_man == 0)
        prod_is_zero.next = (prod_exp == 0) and (prod_man == 0)
        acc_is_nan.next = (acc_exp == 2**EXP_BITS - 1) and (acc_man != 0)
        prod_is_nan.next = (prod_exp == 2**EXP_BITS - 1) and (prod_man != 0)

        # Extended mantissas with implied '1'
        if acc_exp == 0:  # Subnormal or zero
            acc_man_ext.next = acc_man
        else:
            acc_man_ext.next = (1 << MAN_BITS) | acc_man

        if prod_exp == 0:  # Subnormal or zero
            prod_man_ext.next = prod_man
        else:
            prod_man_ext.next = (1 << MAN_BITS) | prod_man

    @always_comb
    def align_operands():
        # Determine exponent difference
        exp_diff = 0

        if acc_is_zero:
            aligned_acc.next = 0
            aligned_prod.next = prod_man_ext << MAN_BITS
        elif prod_is_zero:
            aligned_acc.next = acc_man_ext << MAN_BITS
            aligned_prod.next = 0
        else:
            # Align mantissas based on exponent difference
            if acc_exp > prod_exp:
                exp_diff = acc_exp - prod_exp
                if (
                    exp_diff > 2 * MAN_BITS + EXP_BITS
                ):  # If difference too large, smaller value is negligible
                    aligned_acc.next = acc_man_ext << MAN_BITS
                    aligned_prod.next = 0
                else:
                    aligned_acc.next = acc_man_ext << MAN_BITS
                    aligned_prod.next = prod_man_ext >> exp_diff
            else:
                exp_diff = prod_exp - acc_exp
                if (
                    exp_diff > 2 * MAN_BITS + EXP_BITS
                ):  # If difference too large, smaller value is negligible
                    aligned_acc.next = 0
                    aligned_prod.next = prod_man_ext << MAN_BITS
                else:
                    aligned_acc.next = acc_man_ext >> exp_diff
                    aligned_prod.next = prod_man_ext << MAN_BITS

    @always_seq(clk.posedge, reset=reset)
    def accumulate():
        if reset:
            acc.next = 0
            overflow.next = False
        elif enable:
            # Handle special cases
            if acc_is_nan or prod_is_nan:
                acc.next = (
                    (1 << (FP8_WIDTH - 1)) | ((2**EXP_BITS - 1) << MAN_BITS) | 1
                )  # NaN
                overflow.next = False
            elif acc_is_zero:
                acc.next = product
                overflow.next = False
            elif prod_is_zero:
                # Keep accumulator unchanged
                overflow.next = False
            else:
                # Determine common exponent
                common_exp = max(acc_exp, prod_exp)

                # Add or subtract aligned mantissas based on signs
                if acc_sign == prod_sign:
                    # Same sign - add
                    sum_val.next = aligned_acc + aligned_prod
                    final_sign.next = acc_sign
                else:
                    # Different signs - subtract
                    if aligned_acc >= aligned_prod:
                        sum_val.next = aligned_acc - aligned_prod
                        final_sign.next = acc_sign
                    else:
                        sum_val.next = aligned_prod - aligned_acc
                        final_sign.next = prod_sign

                # Normalize result
                leading_bit_pos = MAN_BITS * 2
                # Find position of leading 1 (simplified)
                # Real hardware would use a leading zero counter
                tmp_sum = sum_val
                if tmp_sum != 0:
                    while (
                        tmp_sum & (1 << leading_bit_pos)
                    ) == 0 and leading_bit_pos > 0:
                        leading_bit_pos -= 1

                # Calculate exponent adjustment
                exp_adj = leading_bit_pos - MAN_BITS

                # Adjust exponent
                adjusted_exp = common_exp - exp_adj

                if sum_val == 0:
                    # Result is zero
                    acc.next = 0
                    overflow.next = False
                elif adjusted_exp >= 2**EXP_BITS - 1:
                    # Overflow - clamp to max value
                    acc.next = (
                        (final_sign << (FP8_WIDTH - 1))
                        | ((2**EXP_BITS - 2) << MAN_BITS)
                        | (2**MAN_BITS - 1)
                    )
                    overflow.next = True
                elif adjusted_exp <= 0:
                    # Underflow - denormalize
                    shift = 1 - adjusted_exp
                    if shift > 2 * MAN_BITS:
                        # Complete underflow
                        acc.next = 0
                    else:
                        # Denormalized result
                        denorm_man = sum_val >> shift
                        if denorm_man < (1 << MAN_BITS):
                            acc.next = (final_sign << (FP8_WIDTH - 1)) | denorm_man
                        else:
                            acc.next = (final_sign << (FP8_WIDTH - 1)) | (
                                2**MAN_BITS - 1
                            )
                    overflow.next = False
                else:
                    # Normal case
                    normalized_man = (sum_val << exp_adj) >> MAN_BITS
                    final_man.next = normalized_man & (2**MAN_BITS - 1)
                    final_exp.next = adjusted_exp
                    acc.next = (
                        (final_sign << (FP8_WIDTH - 1))
                        | (final_exp << MAN_BITS)
                        | final_man
                    )
                    overflow.next = False

    # Connect internal accumulator to output
    @always_comb
    def output_logic():
        result.next = acc

    return instances()
