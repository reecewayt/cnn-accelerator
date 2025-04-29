#!/usr/bin/env python3
"""
Helper script to extract column vectors from matrix A and row vectors from matrix B
for MyHDL test framework. Each vector is converted to a MyHDL bit vector.
"""

from myhdl import *
import numpy as np


def extract_matrix_vectors(A, B, data_width=8):
    """
    Extract column vectors from matrix A and row vectors from matrix B.
    Convert each vector to a MyHDL bit vector.

    Args:
        A: NumPy array or 2D list representing matrix A
        B: NumPy array or 2D list representing matrix B
        data_width: Bit width of each matrix element (default: 8)

    Returns:
        a_vector_list: List of column vectors from A as MyHDL bit vectors
        b_vector_list: List of row vectors from B as MyHDL bit vectors
    """
    # Convert to NumPy arrays if they aren't already
    A_np = np.array(A) if not isinstance(A, np.ndarray) else A
    B_np = np.array(B) if not isinstance(B, np.ndarray) else B

    # Get dimensions
    rows_A, cols_A = A_np.shape
    rows_B, cols_B = B_np.shape

    assert (
        cols_A == rows_B
    ), "Matrix A columns must match Matrix B rows for multiplication."

    # Extract column vectors from A
    a_vector_list = []
    for j in range(cols_A):
        column = A_np[:, j]  # Extract column j
        # Create bit vector for this column
        a_vector = intbv(0)[rows_A * data_width : 0]  # Note the correct range syntax
        for i, val in enumerate(column):
            # With intbv, the slice is [high:low], where high > low
            high_bit = (i + 1) * data_width
            low_bit = i * data_width
            a_vector[high_bit:low_bit] = int(val)
        a_vector_list.append(a_vector)

    # Extract row vectors from B
    b_vector_list = []
    for i in range(rows_B):
        row = B_np[i, :]  # Extract row i
        # Create bit vector for this row
        b_vector = intbv(0)[cols_B * data_width : 0]  # Note the correct range syntax
        for j, val in enumerate(row):
            high_bit = (j + 1) * data_width
            low_bit = j * data_width
            b_vector[high_bit:low_bit] = int(val)
        b_vector_list.append(b_vector)

    return a_vector_list, b_vector_list


def print_bit_vector(name, bit_vector, data_width=8):
    """Print the contents of a bit vector for debugging."""
    print(f"{name} (length: {len(bit_vector)})")
    elements = len(bit_vector) // data_width
    values = []
    for i in range(elements):
        high_bit = (i + 1) * data_width
        low_bit = i * data_width
        val = bit_vector[high_bit:low_bit]
        values.append(val)
    print(f"  Values: {values}")


# Example usage
if __name__ == "__main__":
    # Example matrices
    A = np.array([[5, 7], [1, 2]])  # 2x2 matrix
    B = np.array([[1, 2], [5, 3]])  # 2x2 matrix

    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)

    # Extract vectors
    a_vectors, b_vectors = extract_matrix_vectors(A, B)

    print("Raw Vectors")
    print(a_vectors)
    print(b_vectors)
    print(a_vectors[0])
    print(a_vectors[1])
    print(b_vectors[0])
    print(b_vectors[1])
    print("\nExtracted column vectors from A:")
    for i, vec in enumerate(a_vectors):
        print_bit_vector(f"Column {i}", vec)

    print("\nExtracted row vectors from B:")
    for i, vec in enumerate(b_vectors):
        print_bit_vector(f"Row {i}", vec)

    # Test with matrix multiplication
    C = A @ B  # NumPy matrix multiplication
    print("\nExpected result matrix C = A @ B:")
    print(C)
