def convert_to_e4m3(number):
    """
    Convert a decimal number to E4M3 (8-bit) floating point representation.

    Args:
        number: The decimal number to convert

    Returns:
        tuple: (binary_string, explanation_text)
    """
    # Handle special cases
    if number == 0:
        return "00000000", "Zero is represented as 00000000 in E4M3 format."

    # Determine sign bit
    sign_bit = "1" if number < 0 else "0"
    abs_num = abs(number)

    # Handle NaN (Special case when a number is not a number)
    if abs_num != abs_num:  # NaN check
        return (
            sign_bit + "1111111",
            f"NaN is represented with exponent=1111 and mantissa=111 in E4M3 format: {sign_bit}1111111.",
        )

    # Find the exponent and normalized mantissa
    exponent = 0
    normalized_num = abs_num

    # Normalize the number to be between 1 and 2
    if normalized_num >= 2:
        while normalized_num >= 2:
            normalized_num /= 2
            exponent += 1
    elif normalized_num < 1 and normalized_num > 0:
        while normalized_num < 1:
            normalized_num *= 2
            exponent -= 1

    # Calculate biased exponent (bias is 7 for E4M3)
    biased_exponent = exponent + 7

    # Handle underflow cases
    if biased_exponent < 0:
        # Denormalized number handling (for very small numbers)
        mantissa_shift = biased_exponent
        biased_exponent = 0
        normalized_num = abs_num / (2**-6)  # Scale to the denormalized range

    # Handle values beyond representable range
    # Maximum normal exponent is 14 (1110) - exponent 15 (1111) is for NaN
    if biased_exponent > 14:
        # If we would overflow to exponent 15, we need to check:
        # - If mantissa would be 000, we can use 01111000 (as it's not NaN)
        # - If mantissa would be 001-110, we can use those values (as they're not NaN)
        # - If mantissa would be 111, we need to saturate to 01111110

        if abs_num >= 448.0:  # Would result in 01111111 (NaN)
            return (
                sign_bit + "1111110",
                f"Value {number} would overflow to NaN in E4M3, saturated to {sign_bit}1111110.",
            )
        else:
            # We can represent it with exponent 15 and appropriate mantissa
            biased_exponent = 15
            # Continue with normal mantissa calculation

    # Extract mantissa bits
    if biased_exponent == 0:  # Denormalized
        mantissa_value = normalized_num
    else:  # Normalized
        mantissa_value = normalized_num - 1  # Remove the leading 1 (implicit)

    # Convert mantissa to binary (get 3 bits)
    mantissa_bits = ""
    for i in range(3):
        mantissa_value *= 2
        if mantissa_value >= 1:
            mantissa_bits += "1"
            mantissa_value -= 1
        else:
            mantissa_bits += "0"

    # Round the mantissa (simple round-to-nearest)
    if mantissa_value >= 0.5:
        # Need to round up
        mantissa_as_int = int(mantissa_bits, 2) + 1
        if mantissa_as_int > 7:  # Overflow in mantissa
            mantissa_as_int = 0
            biased_exponent += 1
            # Check for exponent overflow after rounding
            if biased_exponent > 14:
                if biased_exponent == 15 and mantissa_as_int == 7:  # Would be NaN
                    return (
                        sign_bit + "1111110",
                        f"Value {number} would round to NaN in E4M3, saturated to {sign_bit}1111110.",
                    )
                # Otherwise it's representable with exponent 15
                biased_exponent = 15
        mantissa_bits = format(mantissa_as_int, "03b")

    # Special case check: exponent 15 + mantissa 111 = NaN
    if biased_exponent == 15 and mantissa_bits == "111":
        # Saturate to avoid NaN
        mantissa_bits = "110"

    # Format the exponent as 4 bits
    exponent_bits = format(biased_exponent, "04b")

    # Combine to form the 8-bit representation
    binary_repr = sign_bit + exponent_bits + mantissa_bits

    # Create explanation
    if biased_exponent == 0:  # Denormalized
        mantissa_decimal = 0
        for i, bit in enumerate(mantissa_bits):
            if bit == "1":
                mantissa_decimal += 2 ** -(i + 1)

        explanation = f"""
Binary representation: {binary_repr}
- Sign bit (S): {sign_bit} ({'negative' if sign_bit == '1' else 'positive'})
- Exponent bits (E): {exponent_bits} = {biased_exponent} (denormalized form, actual exponent is -6)
- Mantissa bits (M): {mantissa_bits} = {mantissa_decimal:.6f} in decimal

Calculation:
v = (-1)^{sign_bit} × (0 + {mantissa_decimal:.6f}) × 2^(-6)
v = {-1 if sign_bit == '1' else 1} × {mantissa_decimal:.6f} × {2 ** -6:.8f}
v = {-1 if sign_bit == '1' else 1} × {mantissa_decimal * (2 ** -6):.8f}
v = {number}
"""
    else:  # Normalized
        mantissa_decimal = 0
        for i, bit in enumerate(mantissa_bits):
            if bit == "1":
                mantissa_decimal += 2 ** -(i + 1)

        explanation = f"""
Binary representation: {binary_repr}
- Sign bit (S): {sign_bit} ({'negative' if sign_bit == '1' else 'positive'})
- Exponent bits (E): {exponent_bits} = {biased_exponent} (unbiased: {biased_exponent - 7})
- Mantissa bits (M): {mantissa_bits} = {mantissa_decimal:.6f} in decimal

Calculation:
v = (-1)^{sign_bit} × (1 + {mantissa_decimal:.6f}) × 2^({biased_exponent - 7})
v = {-1 if sign_bit == '1' else 1} × {1 + mantissa_decimal:.6f} × {2 ** (biased_exponent - 7):.8f}
v = {-1 if sign_bit == '1' else 1} × {(1 + mantissa_decimal) * (2 ** (biased_exponent - 7)):.8f}
v = {number}
"""

    return binary_repr, explanation


def main():
    print("E4M3 Floating Point Converter")
    print("-----------------------------")
    print("Enter decimal number(s) separated by commas:")

    input_str = input("> ")
    input_values = [float(x.strip()) for x in input_str.split(",")]

    for value in input_values:
        binary, explanation = convert_to_e4m3(value)
        print(f"\nDecimal: {value}")
        print(explanation)
        print("-" * 50)

        # Convert binary to hex for easier reference
        hex_repr = format(int(binary, 2), "02x")
        print(f"Hex representation: 0x{hex_repr}")
        print("-" * 50)


if __name__ == "__main__":
    main()
