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

    # Handle NaN (you might want to define when to return NaN based on your requirements)
    if abs_num != abs_num:  # NaN check
        return (
            sign_bit + "1111000",
            f"NaN is represented as {sign_bit}1111xxx in E4M3 format, where xxx is non-zero.",
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

    # Handle overflow/underflow
    if biased_exponent > 14:  # Maximum representable exponent (all 1's would be NaN)
        return (
            sign_bit + "1110111",
            f"Number too large for E4M3, saturated to max value.",
        )

    if biased_exponent < 0:
        # Denormalized number handling (for very small numbers)
        # In denormalized form, exponent is fixed at -6 (encoded as 0000)
        # and we adjust the mantissa
        mantissa_shift = biased_exponent
        biased_exponent = 0
        normalized_num = abs_num / (2**-6)  # Scale the number to the denormalized range
    else:
        mantissa_shift = 0

    # Extract mantissa bits (normalized_num is between 1 and 2)
    # For normalized numbers, the leading 1 is implicit
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

    # Format the exponent as 4 bits
    exponent_bits = format(biased_exponent, "04b")

    # Combine to form the 8-bit representation
    binary_repr = sign_bit + exponent_bits + mantissa_bits

    # Create explanation
    if biased_exponent == 0:  # Denormalized
        explanation = f"""
Binary representation: {binary_repr}
- Sign bit (S): {sign_bit} ({'negative' if sign_bit == '1' else 'positive'})
- Exponent bits (E): {exponent_bits} = {biased_exponent} (denormalized form, actual exponent is -6)
- Mantissa bits (M): {mantissa_bits} = {mantissa_value:.6f} in decimal

Calculation:
v = (-1)^{sign_bit} × (0 + 0.{mantissa_bits}) × 2^(-6)
v = {-1 if sign_bit == '1' else 1} × {float('0.' + mantissa_bits.replace('', '0', 1))} × {2 ** -6:.8f}
v = {-1 if sign_bit == '1' else 1} × {float('0.' + mantissa_bits.replace('', '0', 1)) * (2 ** -6):.8f}
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


if __name__ == "__main__":
    main()
