def fp8_to_float(fp8_val):
    """Convert an E4M3 value to Python float for verification."""
    # Extract components
    sign = (fp8_val >> 7) & 0x1
    exp = (fp8_val >> 3) & 0xF
    frac = fp8_val & 0x7

    # Handle special cases
    if exp == 0xF:  # All 1's in exponent
        if frac == 0:
            return float("-inf") if sign else float("inf")
        else:
            return float("nan")
    elif exp == 0 and frac == 0:
        return -0.0 if sign else 0.0

    # Normal or denormal processing
    if exp == 0:  # Denormal
        normalized_frac = frac / 8.0
        unbiased_exp = -6  # -7 + 1
    else:  # Normal
        normalized_frac = (frac / 8.0) + 1.0
        unbiased_exp = exp - 7  # Remove bias

    # Calculate value
    value = normalized_frac * (2.0**unbiased_exp)
    return -value if sign else value


def float_to_fp8(f):
    """Convert a Python float to E4M3 format for testing."""
    if f == 0:
        return 0

    # Handle special values
    if f == float("inf"):
        return 0x78  # 0111.1000
    if f == float("-inf"):
        return 0xF8  # 1111.1000
    if f != f:  # NaN check
        return 0x7F  # 0111.1111

    # Handle regular values
    sign = 0x80 if f < 0 else 0
    f = abs(f)

    # Find exponent
    exp = 0
    while f >= 2.0:
        f /= 2.0
        exp += 1
    while f < 1.0 and exp > -7:
        f *= 2.0
        exp -= 1

    # Handle denormals
    if exp <= -7:
        # Denormal number
        frac = int(round(f * 8))
        return sign | frac

    # Normal numbers
    biased_exp = exp + 7
    frac = int(round((f - 1.0) * 8))

    # Handle overflow
    if biased_exp > 14:
        return sign | 0x78  # Infinity with sign

    return sign | (biased_exp << 3) | frac
