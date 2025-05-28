"""
Test for the refactored 3x3 Integer Processing Array
Following the exact pattern from the working 2x2 test
"""

import unittest
from myhdl import *
import numpy as np
import sys
import os

# Import your module and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.hdl.components.processing_array_3x3 import processing_array_3x3
from tests.utils.hdl_test_utils import test_runner
from tests.utils.hdl_bit_vector_helpers import extract_matrix_vectors


class Test3x3ProcessingArray(unittest.TestCase):
    """Test case for the refactored 3x3 Integer Processing Array module."""

    def setUp(self):
        """Set up common signals and parameters for the tests."""
        # Parameters
        self.sim = None
        self.rows = 3
        self.cols = 3
        self.data_width = 8
        self.acc_width = 32

        # Define test matrices (3x3) - small values to avoid overflow
        self.matrix_A = np.array([[2, 1, 3], [1, 2, 1], [3, 1, 2]])  # 3x3 matrix

        self.matrix_B = np.array([[1, 2, 1], [2, 1, 3], [1, 3, 2]])  # 3x3 matrix

        # Calculate expected result using NumPy
        self.expected_C = np.matmul(self.matrix_A, self.matrix_B)

        # Common signals
        self.clk = Signal(bool(0))
        self.reset = ResetSignal(0, active=1, isasync=False)

        # Column Vector of Length Rows from Matrix A
        self.i_a_vector = Signal(intbv(0)[self.rows * self.data_width : 0])

        # Row Vector of Length Cols from Matrix B
        self.i_b_vector = Signal(intbv(0)[self.cols * self.data_width : 0])

        # Control signals
        self.i_data_valid = Signal(bool(0))
        self.i_read_enable = Signal(bool(0))
        self.i_clear_acc = Signal(bool(0))

        # Output signals
        self.o_result_matrix = Signal(
            intbv(0)[self.rows * self.cols * self.acc_width : 0]
        )
        self.o_computation_done = Signal(bool(0))
        self.o_ready_for_data = Signal(bool(0))
        self.o_overflow_detected = Signal(bool(0))

        # Extract vectors using the utility function - EXACTLY like the 2x2 test
        self.a_vectors, self.b_vectors = extract_matrix_vectors(
            self.matrix_A, self.matrix_B, self.data_width
        )

    def tearDown(self):
        if self.sim is not None:
            self.sim.quit()

    def create_3x3_processing_array(self):
        """Helper to create the 3x3 processing array instance."""
        return processing_array_3x3(
            clk=self.clk,
            i_reset=self.reset,
            i_a_vector=self.i_a_vector,
            i_b_vector=self.i_b_vector,
            i_data_valid=self.i_data_valid,
            i_read_enable=self.i_read_enable,
            i_clear_acc=self.i_clear_acc,
            o_result_matrix=self.o_result_matrix,
            o_computation_done=self.o_computation_done,
            o_ready_for_data=self.o_ready_for_data,
            o_overflow_detected=self.o_overflow_detected,
        )

    def testMatrixMultiplication(self):
        """Test basic 3x3 matrix multiplication using the processing array - following 2x2 pattern."""

        @instance
        def test_sequence():
            # Reset the array before starting
            self.reset.next = True
            yield self.clk.posedge
            self.reset.next = False
            yield self.clk.posedge

            # Clear accumulators initially
            print("\n=== Clearing accumulators ===")
            self.i_clear_acc.next = True
            yield self.clk.posedge
            self.i_clear_acc.next = False
            yield self.clk.posedge

            # Print matrices for debugging
            print("\nMatrix A:")
            print(self.matrix_A)
            print("\nMatrix B:")
            print(self.matrix_B)
            print(f"\nExpected result matrix C (A × B):")
            print(self.expected_C)

            # Process vectors in reverse order (last to first)
            # For a 3x3 matrix, this means we'll process column 2, then column 1, then column 0
            # For matrix B, we'll process row 2, then row 1, then row 0
            # EXACTLY following the 2x2 test pattern
            for i in range(len(self.a_vectors) - 1, -1, -1):
                # Get the corresponding index for B vectors (same direction)
                b_idx = i

                print(f"\n=== Processing vector pair {i} ===")
                print(f"A vector {i}: 0x{int(self.a_vectors[i]):06x}")
                print(f"B vector {b_idx}: 0x{int(self.b_vectors[b_idx]):06x}")

                # Assign input vectors
                self.i_a_vector.next = self.a_vectors[i]
                self.i_b_vector.next = self.b_vectors[b_idx]

                # Set data valid and process
                self.i_data_valid.next = True
                yield self.clk.posedge
                self.i_data_valid.next = False

                # Wait a cycle for processing
                yield self.clk.posedge

            # Wait for computation to complete
            for _ in range(3):
                yield self.clk.posedge

            # Enable reading the result
            self.i_read_enable.next = True
            yield self.clk.posedge
            self.i_read_enable.next = False

            # Verify results
            for i in range(self.rows):
                for j in range(self.cols):
                    # Extract result from flattened output
                    index = (i * self.cols + j) * self.acc_width
                    result = int(self.o_result_matrix[index + self.acc_width : index])
                    expected = self.expected_C[i][j]

                    # Print debug info
                    print(f"C[{i}][{j}] = {result}, Expected: {expected}")

                    # Assert equality
                    self.assertEqual(
                        result,
                        expected,
                        f"Result at position ({i},{j}) is {result}, expected {expected}",
                    )

            # Check for overflow (should be False for this test)
            self.assertEqual(
                self.o_overflow_detected, False, f"Overflow detected when not expected"
            )

            print("\n=== All tests passed! ===")

        # Run simulation using the test runner
        self.sim = test_runner(
            self.create_3x3_processing_array,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="processing_array_3x3",
            vcd_output=True,
            verilog_output=False,
            duration=2000,  # Increased duration for 3x3
        )

    def testClearAccumulator(self):
        """Test clearing accumulator functionality - following 2x2 pattern."""

        @instance
        def test_sequence():
            # Reset the array
            self.reset.next = True
            yield self.clk.posedge
            self.reset.next = False
            yield self.clk.posedge

            # Create simple test vectors
            simple_A = np.ones((3, 3), dtype=int)
            simple_B = np.ones((3, 3), dtype=int)

            # Extract vectors
            a_vectors, b_vectors = extract_matrix_vectors(
                simple_A, simple_B, self.data_width
            )

            # Perform one computation cycle
            for i in range(len(a_vectors) - 1, -1, -1):
                b_idx = i

                # Assign input vectors
                self.i_a_vector.next = a_vectors[i]
                self.i_b_vector.next = b_vectors[b_idx]

                # Set data valid and process
                self.i_data_valid.next = True
                yield self.clk.posedge
                self.i_data_valid.next = False

                # Wait a cycle for processing
                yield self.clk.posedge

            # Wait for computation
            for _ in range(3):
                yield self.clk.posedge

            # Clear accumulators
            self.i_clear_acc.next = True
            yield self.clk.posedge
            self.i_clear_acc.next = False
            yield self.clk.posedge

            # Read results (should be zero)
            self.i_read_enable.next = True
            yield self.clk.posedge
            self.i_read_enable.next = False
            yield self.clk.posedge

            # Verify all results are zero
            for i in range(self.rows):
                for j in range(self.cols):
                    # Extract result from flattened output
                    index = (i * self.cols + j) * self.acc_width
                    result = int(self.o_result_matrix[index + self.acc_width : index])

                    self.assertEqual(
                        result,
                        0,
                        f"Expected 0 after clear, got {result} at position ({i},{j})",
                    )

            print("Clear accumulator test passed!")

        # Run simulation
        self.sim = test_runner(
            self.create_3x3_processing_array,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="processing_array_3x3_clear_test",
            vcd_output=True,
            duration=2000,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
