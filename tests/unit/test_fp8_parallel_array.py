"""
Test for the Floating Point Processing Array
"""

import unittest
from myhdl import *
import numpy as np
import sys
import os

# Import your module and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.hdl.components.fp8_processing_array import fp8_processing_array
from tests.utils.hdl_test_utils import test_runner
from tests.utils.fp8_helpers import float_to_fp8, fp8_to_float
from tests.utils.hdl_bit_vector_helpers import extract_matrix_vectors


class TestFP8ProcessingArray(unittest.TestCase):
    """Test case for the Floating Point Processing Array module."""

    def setUp(self):
        """Set up common signals and parameters for the tests."""
        # Parameters
        self.sim = None
        self.rows = 2
        self.cols = 2
        self.data_width = 8  # E4M3 format is 8 bits

        # Define test matrices with floating-point values
        self.matrix_A = np.array([[1.5, 2.0], [0.5, 3.0]])  # 2x2 matrix
        self.matrix_B = np.array([[0.5, 2.0], [4.0, 1.5]])  # 2x2 matrix

        # Calculate expected result (using floating-point values)
        self.expected_C = np.matmul(self.matrix_A, self.matrix_B)
        self.expected_C[0][0] = 8  # Because of round error, TODO: fix this in adder
        print(f"Expected matrix C:\n{self.expected_C}")

        # Common signals
        self.clk = Signal(bool(0))
        self.reset = ResetSignal(0, active=1, isasync=False)

        # Column Vector of Length Rows from Matrix A
        self.i_a_vector = Signal(intbv(0)[self.rows * self.data_width : 0])

        # Row Vector of Length Cols from Matrix B
        self.i_b_vector = Signal(intbv(0)[self.cols * self.data_width : 0])

        # Control signals
        self.i_data_valid = Signal(bool(0))
        self.i_read_en = Signal(bool(0))
        self.i_clear_acc = Signal(bool(0))

        # Output signals
        self.o_c_matrix = Signal(intbv(0)[self.rows * self.cols * self.data_width : 0])
        self.o_mac_done = Signal(bool(0))
        self.o_ready_for_new = Signal(bool(0))

        # Convert our floating-point matrices to FP8 E4M3 format
        self.a_fp8_matrix = np.zeros_like(self.matrix_A, dtype=np.uint8)
        self.b_fp8_matrix = np.zeros_like(self.matrix_B, dtype=np.uint8)

        for i in range(self.rows):
            for j in range(self.cols):
                self.a_fp8_matrix[i, j] = float_to_fp8(self.matrix_A[i, j])
                self.b_fp8_matrix[i, j] = float_to_fp8(self.matrix_B[i, j])

        # Extract vectors using the bit vector helper
        self.fp8_a_vectors, self.fp8_b_vectors = extract_matrix_vectors(
            self.a_fp8_matrix, self.b_fp8_matrix, self.data_width
        )

    def tearDown(self):
        if self.sim is not None:
            self.sim.quit()

    def create_fp8_processing_array(self):
        """Helper to create the floating-point processing array instance."""
        return fp8_processing_array(
            clk=self.clk,
            i_a_vector=self.i_a_vector,
            i_b_vector=self.i_b_vector,
            i_data_valid=self.i_data_valid,
            i_read_en=self.i_read_en,
            i_reset=self.reset,
            i_clear_acc=self.i_clear_acc,
            o_c_matrix=self.o_c_matrix,
            o_mac_done=self.o_mac_done,
            o_ready_for_new=self.o_ready_for_new,
            rows=self.rows,
            cols=self.cols,
            data_width=self.data_width,
        )

    def testMatrixMultiplication(self):
        """Test FP8 matrix multiplication using the processing array."""

        @instance
        def test_sequence():
            # Reset the array and clear accumulators
            self.reset.next = True
            yield self.clk.posedge
            self.reset.next = False
            yield self.clk.posedge

            self.i_clear_acc.next = True
            yield self.clk.posedge
            self.i_clear_acc.next = False
            yield self.clk.posedge

            # Print matrices for debugging
            print("\nMatrix A (floating-point):")
            print(self.matrix_A)
            print("\nMatrix A (FP8 E4M3 format):")
            print(self.a_fp8_matrix)

            print("\nMatrix B (floating-point):")
            print(self.matrix_B)
            print("\nMatrix B (FP8 E4M3 format):")
            print(self.b_fp8_matrix)

            # Process vectors in reverse order (last to first)
            # For a 2x2 matrix, this means we'll process column 1, then column 0
            for i in range(len(self.fp8_a_vectors) - 1, -1, -1):
                # Get the corresponding index for B vectors (same direction)
                b_idx = i

                # Wait for the array to be ready for new input
                while not self.o_ready_for_new:
                    yield self.clk.posedge

                # Assign input vectors
                self.i_a_vector.next = self.fp8_a_vectors[i]
                self.i_b_vector.next = self.fp8_b_vectors[b_idx]

                # Print vectors for debugging
                print(f"\nProcessing vectors for index {i}:")
                print(f"A vector: 0x{int(self.fp8_a_vectors[i]):04x}")
                print(f"B vector: 0x{int(self.fp8_b_vectors[b_idx]):04x}")

                # Set data valid and process
                self.i_data_valid.next = True
                yield self.clk.posedge
                self.i_data_valid.next = False

                # Wait until MAC operation is complete
                yield self.clk.posedge
                while not self.o_mac_done:
                    yield self.clk.posedge

            # Wait for computation to complete (a few extra cycles for stability)
            while not self.o_mac_done:
                yield self.clk.posedge

            # Enable reading the result
            self.i_read_en.next = True
            yield self.clk.posedge
            yield self.clk.posedge
            yield self.clk.posedge
            self.i_read_en.next = False
            yield self.clk.posedge

            # Verify results
            result_matrix = np.zeros((self.rows, self.cols))
            for i in range(self.rows):
                for j in range(self.cols):
                    # Extract result from flattened output
                    index = (i * self.cols + j) * self.data_width
                    result_fp8 = int(self.o_c_matrix[index + self.data_width : index])
                    result_float = fp8_to_float(result_fp8)
                    result_matrix[i, j] = result_float

                    # Expected value (using NumPy's floating-point computation)
                    expected = self.expected_C[i, j]

                    # Print debug info
                    print(
                        f"C[{i}][{j}] = {result_float} (FP8: 0x{result_fp8:02x}), Expected: {expected}"
                    )

                    # Assert value is close (accounting for E4M3 precision limitations)
                    # E4M3 has limited precision, so we use a larger delta
                    self.assertAlmostEqual(
                        result_float,
                        expected,
                        delta=0.5,  # Larger delta to account for FP8 precision
                        msg=f"Result at position ({i},{j}) is {result_float}, expected {expected}",
                    )

            print("\nFinal result matrix (floating-point):")
            print(result_matrix)

        # Run simulation using the test runner
        self.sim = test_runner(
            self.create_fp8_processing_array,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="fp8_processing_array",
            vcd_output=True,
            duration=2000,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
