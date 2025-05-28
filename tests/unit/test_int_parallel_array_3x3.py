"""
Test for the hardcoded 3x3 Integer Processing Array
"""

import unittest
from myhdl import *
import numpy as np
import sys
import os

# Import your module and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from hdl.components.int_processing_array import processing_array_3x3
from tests.utils.hdl_test_utils import test_runner


class Test3x3ProcessingArray(unittest.TestCase):
    """Test case for the 3x3 Integer Processing Array module."""

    def setUp(self):
        """Set up common signals and parameters for the tests."""
        self.sim = None

        # Hardcoded for 3x3 design
        self.array_size = 3
        self.data_width = 8
        self.acc_width = 16

        # Define test matrices (3x3)
        self.matrix_A = np.array([[2, 3, 1], [4, 1, 2], [1, 2, 3]], dtype=np.int8)

        self.matrix_B = np.array([[1, 2, 1], [3, 1, 2], [2, 1, 4]], dtype=np.int8)

        # Calculate expected result using NumPy
        self.expected_C = np.matmul(self.matrix_A, self.matrix_B)
        print(f"Expected matrix C (A × B):\n{self.expected_C}")

        # Create signals
        self.clk = Signal(bool(0))
        self.reset = ResetSignal(0, active=1, isasync=False)

        # Input vectors (24 bits each for 3 × 8-bit elements)
        self.i_a_vector = Signal(intbv(0)[24:])
        self.i_b_vector = Signal(intbv(0)[24:])

        # Control signals
        self.i_data_valid = Signal(bool(0))
        self.i_read_enable = Signal(bool(0))
        self.i_clear_acc = Signal(bool(0))

        # Output signals (144 bits for 9 × 16-bit elements)
        self.o_result_matrix = Signal(intbv(0)[144:])
        self.o_computation_done = Signal(bool(0))
        self.o_ready_for_data = Signal(bool(0))
        self.o_overflow_detected = Signal(bool(0))

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

    def pack_vector(self, vector, data_width=8):
        """Pack a vector into a bit vector."""
        packed = intbv(0)[len(vector) * data_width :]
        for i, val in enumerate(vector):
            # Convert to unsigned representation for packing
            if val < 0:
                unsigned_val = (1 << data_width) + val  # Two's complement
            else:
                unsigned_val = val
            packed[(i + 1) * data_width : i * data_width] = unsigned_val
        return packed

    def unpack_result_matrix(self, packed_result, num_elements=9, element_width=16):
        """Unpack the result matrix from bit vector."""
        result = np.zeros((3, 3), dtype=np.int16)

        for i in range(3):  # rows
            for j in range(3):  # cols
                element_idx = i * 3 + j
                start_bit = element_idx * element_width
                end_bit = (element_idx + 1) * element_width

                # Extract bits
                raw_val = int(packed_result[end_bit:start_bit])

                # Convert from unsigned to signed if needed
                if raw_val >= (1 << (element_width - 1)):
                    signed_val = raw_val - (1 << element_width)
                else:
                    signed_val = raw_val

                result[i][j] = signed_val

        return result

    def testMatrixMultiplication(self):
        """Test 3x3 matrix multiplication using the processing array."""

        @instance
        def test_sequence():
            # Reset the array
            self.reset.next = True
            yield self.clk.posedge
            yield self.clk.posedge
            self.reset.next = False
            yield self.clk.posedge

            # Clear accumulators initially
            print("\n=== Clearing accumulators ===")
            self.i_clear_acc.next = True
            yield self.clk.posedge
            self.i_clear_acc.next = False
            yield self.clk.posedge

            # Print input matrices for debugging
            print("\nMatrix A:")
            print(self.matrix_A)
            print("\nMatrix B:")
            print(self.matrix_B)

            # Process each column of A with each row of B
            for col in range(self.array_size):
                print(f"\n=== Processing column {col} of A with all rows of B ===")

                # Extract column from A and prepare as vector
                a_column = self.matrix_A[:, col]
                print(f"A column {col}: {a_column}")

                for row in range(self.array_size):
                    print(f"\n--- Processing with row {row} of B ---")

                    # Extract row from B
                    b_row = self.matrix_B[row, :]
                    print(f"B row {row}: {b_row}")

                    # Pack vectors
                    packed_a = self.pack_vector(a_column)
                    packed_b = self.pack_vector(b_row)

                    print(f"Packed A: 0x{int(packed_a):06x}")
                    print(f"Packed B: 0x{int(packed_b):06x}")

                    # Wait for ready
                    timeout = 0
                    while not self.o_ready_for_data and timeout < 100:
                        yield self.clk.posedge
                        timeout += 1

                    if not self.o_ready_for_data:
                        self.fail("Timeout waiting for ready signal")

                    # Set inputs
                    self.i_a_vector.next = packed_a
                    self.i_b_vector.next = packed_b

                    # Start computation
                    self.i_data_valid.next = True
                    yield self.clk.posedge
                    self.i_data_valid.next = False

                    # Wait for computation to complete
                    timeout = 0
                    while not self.o_computation_done and timeout < 100:
                        yield self.clk.posedge
                        timeout += 1

                    if not self.o_computation_done:
                        self.fail("Timeout waiting for computation done")

                    print("Computation completed")

                    # Wait for data_valid to be processed
                    yield self.clk.posedge
                    yield self.clk.posedge

            # Read final results
            print("\n=== Reading final results ===")
            self.i_read_enable.next = True
            yield self.clk.posedge
            yield self.clk.posedge
            self.i_read_enable.next = False
            yield self.clk.posedge

            # Verify results
            result_matrix = self.unpack_result_matrix(self.o_result_matrix)

            print(f"\nFinal result matrix:")
            print(result_matrix)
            print(f"\nExpected result matrix:")
            print(self.expected_C)

            # Check for overflow
            if self.o_overflow_detected:
                print("WARNING: Overflow detected during computation")

            # Verify each element
            for i in range(3):
                for j in range(3):
                    expected = self.expected_C[i, j]
                    actual = result_matrix[i, j]
                    print(f"C[{i}][{j}]: expected={expected}, actual={actual}")

                    self.assertEqual(
                        actual,
                        expected,
                        f"Mismatch at position ({i},{j}): expected {expected}, got {actual}",
                    )

            print("\n=== All tests passed! ===")

        # Run simulation
        self.sim = test_runner(
            self.create_3x3_processing_array,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="processing_array_3x3",
            vcd_output=True,
            verilog_output=False,
            duration=5000,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
