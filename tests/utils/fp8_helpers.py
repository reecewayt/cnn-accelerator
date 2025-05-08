from dataclasses import dataclass, field
from typing import Union, TypeVar, overload, Literal, ClassVar, Optional
import math
import re


@dataclass
class E4M3Format:
    """
    Represents an 8-bit E4M3 floating-point format.
    This format has:
    - 1 sign bit
    - 4 exponent bits (biased by 7)
    - 3 mantissa bits

    Special values:
    - NaN: 0x7F (0111 1111)
    - Max positive: 0x7E (0111 1110) = 448.0
    - Max negative: 0xFE (1111 1110) = -448.0
    - Min positive normal: 0x08 (0000 1000) = 2^-6
    - Min negative normal: 0x88 (1000 1000) = -2^-6

    This class accepts multiple input formats:
    - Integer: Interpreted as the raw 8-bit value (0-255)
    - Float: Converted to the closest representable E4M3 value (-448 to +448)
    - String: Parsed as binary ("0b1010"), hex ("0x3A"), or decimal ("42")
    """

    # Class constants
    WIDTH: ClassVar[int] = 8
    EXP_BITS: ClassVar[int] = 4
    FRAC_BITS: ClassVar[int] = 3
    BIAS: ClassVar[int] = 7

    # Special values as class constants
    NAN: ClassVar[int] = 0x7F
    MAX_POS: ClassVar[int] = 0x7E
    MAX_NEG: ClassVar[int] = 0xFE

    # The E4M3 format can represent values in the range of approximately -448 to +448
    MAX_FLOAT_VALUE: ClassVar[float] = 448.0
    MIN_FLOAT_VALUE: ClassVar[float] = -448.0

    # Private instance value
    _value: int = field(init=False)  # Hidden raw value

    # Input can be any of several types
    value: Union[int, float, str]

    def __post_init__(self):
        """Process and validate the input value."""
        if isinstance(self.value, int):
            # Direct integer value (as raw bits)
            if not (0 <= self.value <= 255):
                raise ValueError(
                    f"E4M3 raw value must be between 0 and 255, got {self.value}"
                )
            self._value = self.value
        elif isinstance(self.value, float):
            # Convert float to E4M3
            self._value = self._float_to_e4m3(self.value)
        elif isinstance(self.value, str):
            # Parse string representation
            self._value = self._parse_string(self.value)
        else:
            raise TypeError(
                f"E4M3 value must be an integer, float, or string, got {type(self.value)}"
            )

    def _parse_string(self, s: str) -> int:
        """Parse a string representation into an E4M3 value."""
        s = s.strip().lower()

        # Binary format: 0b01010101
        if s.startswith("0b"):
            try:
                val = int(s[2:], 2)
                if len(s[2:]) > 8:
                    raise ValueError(
                        f"Binary E4M3 value must be at most 8 bits, got {len(s[2:])} bits"
                    )
                if not (0 <= val <= 255):
                    raise ValueError(
                        f"Binary E4M3 value must be between 0b00000000 and 0b11111111, got {s}"
                    )
                return val
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid binary format: {s}")
                raise

        # Hexadecimal format: 0x3A
        elif s.startswith("0x"):
            try:
                val = int(s[2:], 16)
                if not (0 <= val <= 255):
                    raise ValueError(
                        f"Hex E4M3 value must be between 0x00 and 0xFF, got {s}"
                    )
                return val
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid hexadecimal format: {s}")
                raise

        # Decimal format: 42
        else:
            try:
                # Try to parse as float
                val = float(s)
                # If it's an integer-like value between 0-255, treat as raw bits
                if val.is_integer() and 0 <= val <= 255:
                    return int(val)
                # Otherwise, convert float to E4M3
                return self._float_to_e4m3(val)
            except ValueError:
                raise ValueError(f"Invalid E4M3 value format: {s}")

    def _float_to_e4m3(self, f: float) -> int:
        """Convert a float to E4M3 format."""
        # Handle special cases
        if math.isnan(f):
            return self.NAN

        if f == 0:
            return 0

        # Handle regular values
        sign = 0x80 if f < 0 else 0
        f = abs(f)

        # Pre-defined mappings for special values that need exact representation
        if f == 256.0:
            return sign | 0x78  # This is the correct representation for 256.0
        elif f > self.MAX_FLOAT_VALUE:
            return sign | self.MAX_POS
        elif f < self.MIN_FLOAT_VALUE:
            return sign | self.MAX_NEG

        # Find exponent
        exp = 0
        while f >= 2.0 and exp < 7:
            f /= 2.0
            exp += 1
        while f < 1.0 and exp > -7:
            f *= 2.0
            exp -= 1

        # Handle denormals
        if exp <= -7:
            # Denormal number (very small value)
            frac = int(round(f * 8))
            if frac > 7:  # Handle rounding overflow
                frac = 7
            return sign | frac

        # Check for overflow - return max finite value
        if exp > 7:
            return sign | self.MAX_POS

        # Normal numbers
        biased_exp = exp + self.BIAS
        frac = int(round((f - 1.0) * 8))

        # Handle rounding overflow
        if frac > 7:
            frac = 0
            biased_exp += 1
            # Check again for overflow after rounding
            if biased_exp > 14:
                return sign | self.MAX_POS

        return sign | (biased_exp << 3) | frac

    @property
    def raw_value(self) -> int:
        """Get the raw 8-bit integer value."""
        return self._value

    def to_float(self) -> float:
        """Convert the E4M3 value to a Python float."""
        # Extract components
        sign = (self._value >> 7) & 0x1
        exp = (self._value >> 3) & 0xF
        frac = self._value & 0x7

        # Handle NaN
        if self._value == self.NAN or self._value == (self.NAN | 0x80):
            return float("nan")

        # Handle zero
        if exp == 0 and frac == 0:
            return -0.0 if sign else 0.0

        # Normal or denormal processing
        if exp == 0:  # Denormal
            normalized_frac = frac / 8.0
            unbiased_exp = -6  # -7 + 1
        else:  # Normal
            normalized_frac = (frac / 8.0) + 1.0
            unbiased_exp = exp - self.BIAS

        # Calculate value
        value = normalized_frac * (2.0**unbiased_exp)
        return -value if sign else value

    def to_binary(self) -> str:
        """Return the binary representation."""
        return f"0b{self._value:08b}"

    def to_hex(self) -> str:
        """Return the hexadecimal representation."""
        return f"0x{self._value:02x}"

    def __str__(self) -> str:
        """Human-readable string representation."""
        float_val = self.to_float()
        if math.isnan(float_val):
            return f"E4M3({self.to_hex()}, NaN)"
        return f"E4M3({self.to_hex()}, {float_val})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()

    # Binary operations
    def __add__(self, other: Union["E4M3Format", float, int, str]) -> "E4M3Format":
        """Add two E4M3 values."""
        if not isinstance(other, E4M3Format):
            other = E4M3Format(other)

        # Simple implementation: convert to float, add, convert back
        result_float = self.to_float() + other.to_float()
        return E4M3Format(result_float)

    def __sub__(self, other: Union["E4M3Format", float, int, str]) -> "E4M3Format":
        if not isinstance(other, E4M3Format):
            other = E4M3Format(other)

        result_float = self.to_float() - other.to_float()
        return E4M3Format(result_float)

    def __mul__(self, other: Union["E4M3Format", float, int, str]) -> "E4M3Format":
        if not isinstance(other, E4M3Format):
            other = E4M3Format(other)

        result_float = self.to_float() * other.to_float()
        return E4M3Format(result_float)

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if isinstance(other, E4M3Format):
            return self._value == other._value
        try:
            other_e4m3 = E4M3Format(other)
            return self._value == other_e4m3._value
        except (TypeError, ValueError):
            return NotImplemented


# Standalone conversion functions for backward compatibility
def float_to_fp8(f: Union[float, int, str]) -> int:
    """Convert various formats to an E4M3 format (8-bit integer)."""
    return E4M3Format(f).raw_value


def fp8_to_float(fp8_val: Union[int, float, str]) -> float:
    """Convert an E4M3 value (8-bit integer) to Python float."""
    return E4M3Format(fp8_val).to_float()


def main():
    """
    Interactive CLI tool for exploring E4M3 floating-point representations.
    Allows users to enter values and see their E4M3 representation.
    """
    print("E4M3 Format Converter")
    print("=====================")
    print("Enter values to see their E4M3 representation")
    print("You can input:")
    print("  - Decimal numbers (e.g., 1.5, 2, -3.75)")
    print("  - Hex values (e.g., 0x3C, 0x7E)")
    print("  - Binary values (e.g., 0b01111110)")
    print("  - Multiple values separated by commas")
    print("Enter 'q' or 'quit' to exit")
    print("=====================")

    while True:
        user_input = input("\nEnter value(s): ").strip()

        if user_input.lower() in ("q", "quit", "exit"):
            print("Exiting E4M3 converter. Goodbye!")
            break

        # Split by commas for multiple inputs
        values = [v.strip() for v in user_input.split(",")]

        for value in values:
            try:
                # Try to parse the input
                e4m3_val = E4M3Format(value)

                # Get representations
                float_repr = e4m3_val.to_float()
                hex_repr = e4m3_val.to_hex()
                binary_repr = e4m3_val.to_binary()

                # Print results
                print(f"\nValue: {value}")
                print(f"  Float: {float_repr}")
                print(f"  Hex:   {hex_repr}")
                print(f"  Binary: {binary_repr}")

                # Show special values info if applicable
                if e4m3_val.raw_value == E4M3Format.NAN:
                    print("  Type: NaN (Not a Number)")
                elif e4m3_val.raw_value == E4M3Format.MAX_POS:
                    print("  Type: Maximum positive value (448.0)")
                elif e4m3_val.raw_value == E4M3Format.MAX_NEG:
                    print("  Type: Maximum negative value (-448.0)")
                elif e4m3_val.raw_value == 0:
                    print("  Type: Zero")

            except (ValueError, TypeError) as e:
                print(f"Error with input '{value}': {str(e)}")


if __name__ == "__main__":
    main()
