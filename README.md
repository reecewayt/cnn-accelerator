# cnn-accelerator
A CNN accelerator based in MyHDL (Python)

### Install required packages

```bash
python -m venv venv

source venv/bin/activate

# Install packages from requirements.txt
pip install -r requirements.txt
```


### Project Structure

```bash
cnn-accelerator/
├── src/
│   ├── __init__.py
│   ├── hdl/                  # MyHDL source code
│   │   ├── __init__.py
│   │   ├── layers/           # CNN layer implementations
│   │   ├── memory/           # Memory interfaces
│   │   └── top.py            # Top-level design
│   └── utils/                # Helper functions
│       └── __init__.py
├── gen/                      # Generated Verilog/VHDL files
│   └── verilog/              # Generated Verilog output
├── tests/                    # Test files
│   ├── unit/                 # Unit tests for individual components
│   └── integration/          # Full system tests
├── docs/                     # Documentation
├── scripts/                  # Build scripts, automation
│   └── generate_hdl.py       # Script to run code generation
├── venv/                     # Virtual environment (gitignored)
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

## Mathematical Background

### Floating Point [E4M3](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) Representation
E4M3 is an 8-bit floating point format designed for low-precision machine learning applications. It consists of:
```
[S|EEEE|MMM]
 ↑  ↑    ↑
 |  |    └── Mantissa (3 bits)
 |  └─────── Exponent (4 bits, bias of 7)
 └────────── Sign (1 bit)
```
**Mathematical Representation**
For a given bit pattern `[S|EEEE|MMM]`, the value is calculated as:

$$v = (-1)^S \times (1 + M) \times 2^{(E-7)}$$

Where:
- $S$ is the sign bit (0 for positive, 1 for negative)
- $E$ is the unsigned integer value of the exponent bits
- $M$ is the fraction value of the mantissa bits (interpreted as 0.MMM)
- The bias for E4M3 is 7

**Example Calculation**
For the bit pattern `00101101` representing 0.40625:

- Sign bit ($S$): 0 (positive)
- Exponent bits ($E$): 0101 = 5 in decimal
- Mantissa bits ($M$): .101 = 0.625 in decimal (0.5 + 0.125)

Applying the formula:
$$v = (-1)^0 \times (1 + 0.625) \times 2^{(5-7)}$$
$$v = 1 \times 1.625 \times 2^{-2}$$
$$v = 1.625 \times 0.25$$
$$v = 0.40625$$

This value (0.40625) is the closest representable number to 0.3952 in the E4M3 format due to the limited precision.

**Special Cases:**
E4M3 format allows one special format which is when the exponent bits are all 1's; this represents `NaN`.

**Precision Limitations**
With only 3 bits for the mantissa, E4M3 can only represent 8 different fraction values per exponent, making it suitable for applications where memory efficiency is more important than high precision.

### Floating Point Multiplication
This project includes a FP MAC unit and FP Parallel Processing Unit. Below is the description for multiplication and addition that is at the heart of this design. Remember we are using the E4M3 format.

1. Handle special cases (NaN or Zero):
       - If either input is NaN, result is NaN
       - If (Infinity * 0), result is NaN
       - If either input is Infinity (and other is non-zero), result is NaN
       - If either input is zero, result is zero

3. Process Normal Numbers:
   - For normalized numbers, add implicit '1' bit to mantissa
   - For denormalized numbers, exponent is -6 but no implicit bit

4. Multiply:
   - Result sign = XOR of input signs
   - Result exponent = sum of unbiased exponents
     * Unbias by subtracting 7 from each exponent
     * Add 7 back to final exponent
   - Result mantissa = product of mantissas (with implicit bits)
     * This creates a 8-bit product (from 4-bit × 4-bit)

5. Normalize:
   - If product's MSB is set, shift right and increment exponent
   - If product's MSB is not set, shift left and decrement exponent
   - Extract top 3 bits for mantissa, use 4th bit for rounding

6. Handle Overflow/Underflow:
   - If exponent > 7, result is Infinity
   - If exponent < -6, result is zero or denormalized

7. Round:
   - Use round-to-nearest, ties to even
   - If rounding causes mantissa overflow, increment exponent

8. Pack:
   - Combine sign, exponent, and mantissa into final 8-bit result
   - For special values (0, NaN), use standard encodings:
     * Zero: S.0000.000
     * NaN: S.1111.xxx (non-zero mantissa)

**Why normalization is needed in floating point multiplication**
Normalization is the process of representing a number so that the mantissa has a leading 1 (which is then implicit). E4M3 format can handle very small number by treating numbers with all zeros in the exponent as denormalized.

For example:
- If exponent is `0001` the smallest possible number is $\pm 1.000*2^{-6} = \pm 0.015625$
- If exponent is `0000` we treat it as denormalized and smallest possible value is $\pm 0.001*2^{-6} = \pm 0.0002441...$ this is because there is not implicit one for this special case.

In conclusion, normalization helps with the following:

1. Normalization ensures the mantissa uses its full bit range without normalization, we'd waste bits and lose precision.
        - Example, 0.01 × 2^2 should be normalized to 1.0 × 2^0.

2. Normalization ensure only one representation of a number.
       - Example: 0.1 × 2^1 and 1.0 × 2^0 represent the same value

3. Handles Multiplication Correctly
       - When we multiply mantissas (both with implicit leading 1s),
         the result may exceed the format's range
       - Example: (1.xxx × 2^a) × (1.yyy × 2^b) = (1.xxx × 1.yyy) × 2^(a+b)
       - The product (1.xxx × 1.yyy) could be between 1.0 and just under 4.0
       - If result >= 2.0, we need to shift right and increment the exponent

4. Prevents Underflow/Overflow
       - After multiplication, we may need to adjust the result
       - If product is too small, we shift left (decrement exponent)
       - If product is too large, we shift right (increment exponent)
       - These adjustments ensure we maximize precision while staying in range

**Without normalization, floating-point arithmetic would be less accurate,
harder to implement, and would suffer from comparison inconsistencies.**

### Floating Point Addition
Algorithm for floating point addition in the E4M3 format:

1. **Unpack operands**: Extract the sign bit, exponent bits (4), and mantissa bits (3) from each input.

2. **Handle special cases**: Check for special cases like zeros. If one operand is zero, the result is the other operand.

3. **Align operands**:
   - Determine which number has the smaller exponent
   - Shift the mantissa of the smaller exponent number right by the exponent difference
   - This aligns the binary points so addition can be performed correctly

4. **Perform addition**:
   - If signs are the same, add the aligned mantissas
   - If signs differ, subtract the smaller mantissa from the larger
   - The result sign is determined by the larger operand's sign

5. **Normalize result**:
   - If addition caused an overflow, shift right and increment exponent
   - If subtraction resulted in leading zeros, shift left and decrement exponent
   - This ensures the result maintains the form 1.xxx × 2^exp

6. **Round result**:
   - Apply appropriate rounding (typically round-to-nearest)
   - Handle any additional normalization if rounding causes overflow

7. **Handle overflow/underflow**:
   - If result exceeds representable range, saturate to maximum value
   - If result is too small, flush to zero or smallest denormal

8. **Pack result**: Combine the sign, exponent, and mantissa bits into the final 8-bit result.

The most difficult aspects are mantissa alignment and post-operation normalization, which are critical for preserving precision in the limited bit format.

### Image 2 Col (im2col)
The Image-to-Column (im2col) transformation reorganizes the input tensor into a two-dimensional matrix, with each column representing a flattened region of the input tensor covered by the kernel. This transformation enables the seamless conversion of convolution into standard matrix multiplication
## Direct Convolution Method

Let's use a simple example:
- Input: 4×4 matrix with a single channel
- Kernel: 3×3 with a single channel
- Stride: 1
- No padding

Input image:
```
1  2  3  4
5  6  7  8
9  10 11 12
13 14 15 16
```

Kernel:
```
1  0  1
0  1  0
1  0  1
```

Using direct convolution, we would slide this kernel over the input image, computing dot products at each location. With stride 1 and no padding, we get a 2×2 output.

Output[0,0] = 1×1 + 0×2 + 1×3 + 0×5 + 1×6 + 0×7 + 1×9 + 0×10 + 1×11 = 1+3+6+9+11 = 30

Output[0,1] = 1×2 + 0×3 + 1×4 + 0×6 + 1×7 + 0×8 + 1×10 + 0×11 + 1×12 = 2+4+7+10+12 = 35

Output[1,0] = 1×5 + 0×6 + 1×7 + 0×9 + 1×10 + 0×11 + 1×13 + 0×14 + 1×15 = 5+7+10+13+15 = 50

Output[1,1] = 1×6 + 0×7 + 1×8 + 0×10 + 1×11 + 0×12 + 1×14 + 0×15 + 1×16 = 6+8+11+14+16 = 55

Final output:
```
30 35
50 55
```

## Im2col Method

Now, let's see how im2col transforms this problem:

1. First, we extract each 3×3 patch from the input that the kernel will convolve with, and arrange them as columns.

For our 4×4 input with a 3×3 kernel and stride 1, we get 4 patches (2×2 output locations):

Patch 1 (top-left):
```
1  2  3
5  6  7
9  10 11
```

Patch 2 (top-right):
```
2  3  4
6  7  8
10 11 12
```

Patch 3 (bottom-left):
```
5  6  7
9  10 11
13 14 15
```

Patch 4 (bottom-right):
```
6  7  8
10 11 12
14 15 16
```

2. We reshape each patch into a column vector (reading row by row):

Patch 1: [1, 2, 3, 5, 6, 7, 9, 10, 11]ᵀ
Patch 2: [2, 3, 4, 6, 7, 8, 10, 11, 12]ᵀ
Patch 3: [5, 6, 7, 9, 10, 11, 13, 14, 15]ᵀ
Patch 4: [6, 7, 8, 10, 11, 12, 14, 15, 16]ᵀ

3. Place these columns side by side to form the im2col matrix:

```
1  2  5  6
2  3  6  7
3  4  7  8
5  6  9  10
6  7  10 11
7  8  11 12
9  10 13 14
10 11 14 15
11 12 15 16
```

4. Reshape our kernel into a row vector:

Kernel: [1, 0, 1, 0, 1, 0, 1, 0, 1]

5. Perform matrix multiplication between the kernel vector and the im2col matrix:

[1, 0, 1, 0, 1, 0, 1, 0, 1] ×
```
1  2  5  6
2  3  6  7
3  4  7  8
5  6  9  10
6  7  10 11
7  8  11 12
9  10 13 14
10 11 14 15
11 12 15 16
```
= [30, 35, 50, 55]

6. Reshape the result back to the output dimensions (2×2):

```
30 35
50 55
```

This is exactly the same result as direct convolution!

## Why im2col is Efficient

The key advantage of im2col is that it transforms convolution operations into matrix multiplications, which are highly optimized in modern computing libraries and hardware. For multiple filters/output channels, we can stack multiple kernel row vectors into a matrix and perform a single matrix multiplication.

This approach allows CNNs to leverage highly optimized linear algebra libraries (like BLAS) and makes better use of parallel computing resources (GPUs). The tradeoff is increased memory usage, as the im2col operation duplicates input values across multiple columns.
