import unittest
from myhdl import *
import numpy as np
import sys
import os

# Import your module and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.hdl.components.processing_array import processing_array
from tests.utils.hdl_test_utils import test_runner
from tests.utils.hdl_bit_vector_helpers import extract_matrix_vectors


class TestProcessingArrayUnit(unittest.TestCase):
    """Test case for the Processing Array module."""

    def setUp(self):
        """Set up common signals and parameters for the tests."""
        # Parameters
        self.sim = None
        self.rows = 2
        self.cols = 2
        self.data_width = 8
        self.acc_width = 16

        # Define test matrices
        self.matrix_A = np.array([[5, 7], [1, 2]])  # 2x2 matrix
        self.matrix_B = np.array([[1, 2], [5, 3]])  # 2x2 matrix

        # Calculate expected result
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
        self.i_read_en = Signal(bool(0))

        # Output signals
        self.o_c_matrix = Signal(intbv(0)[self.rows * self.cols * self.acc_width : 0])
        self.o_saturate_detect = Signal(bool(0))

        self.a_vectors, self.b_vectors = extract_matrix_vectors(
            self.matrix_A, self.matrix_B, self.data_width
        )
        print("Raw Vectors")
        print(self.a_vectors)
        print(self.b_vectors)

    def create_processing_array(self):
        """Helper to create the processing array instance."""
        return processing_array(
            clk=self.clk,
            i_a_vector=self.i_a_vector,
            i_b_vector=self.i_b_vector,
            i_data_valid=self.i_data_valid,
            i_read_en=self.i_read_en,
            i_reset=self.reset,
            o_c_matrix=self.o_c_matrix,
            o_saturate_detect=self.o_saturate_detect,
            rows=self.rows,
            cols=self.cols,
            data_width=self.data_width,
            acc_width=self.acc_width,
        )

    def testMatrixMultiplication(self):
        """Test basic matrix multiplication using the processing array."""

        @instance
        def test_sequence():

            # Reset the array before starting
            self.reset.next = True
            yield self.clk.posedge
            self.reset.next = False
            yield self.clk.posedge

            # Process vectors in reverse order (last to first)
            # For a 2x2 matrix, this means we'll process column 1, then column 0
            # For matrix B, we'll process row 1, then row 0
            for i in range(len(self.a_vectors) - 1, -1, -1):
                # Get the corresponding index for B vectors (same direction)
                b_idx = i

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
            self.i_read_en.next = True
            yield self.clk.posedge
            self.i_read_en.next = False

            # Verify results
            for i in range(self.rows):
                for j in range(self.cols):
                    # Extract result from flattened output
                    index = (i * self.cols + j) * self.acc_width
                    result = self.o_c_matrix[index : index + self.acc_width]
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
                self.o_saturate_detect, False, f"Overflow detected when not expected"
            )

        # Run simulation using the test runner
        self.sim = test_runner(
            self.create_processing_array,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="processing_array",
            vcd_output=True,
            duration=500,
        )

    def testOverflow(self):
        """Test overflow detection in the processing array."""

        @instance
        def test_sequence():
            # Create matrices with maximum values
            max_val = 2**self.data_width - 1  # 255 for 8-bit
            overflow_A = np.ones((self.rows, self.cols), dtype=int) * max_val
            overflow_B = np.ones((self.rows, self.cols), dtype=int) * max_val

            # Extract vectors from matrices
            a_vectors, b_vectors = extract_matrix_vectors(
                overflow_A, overflow_B, self.data_width
            )

            # Reset the array before starting
            self.reset.next = True
            yield self.clk.posedge
            self.reset.next = False
            yield self.clk.posedge

            # Process vectors in reverse order (last to first)
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

            # Wait for computation to complete
            for _ in range(3):
                yield self.clk.posedge

            # Enable reading the result
            self.i_read_en.next = True
            yield self.clk.posedge
            self.i_read_en.next = False

            # Verify that overflow was detected
            self.assertEqual(
                self.o_saturate_detect, True, f"Overflow not detected when expected"
            )

        # Run simulation using the test runner
        self.sim = test_runner(
            self.create_processing_array,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="processing_array",
            vcd_output=True,
            duration=500,
        )


if __name__ == "__main__":
    unittest.main()
