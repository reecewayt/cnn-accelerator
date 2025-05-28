"""
Parallel processing array for matrix multiplication, similar to a systolic array.
All inputs are available at the same time, and the outputs are collected in a single clock cycle.
The processing elements (PEs) are arranged in a 2D array, each performing a MAC operation.
Input is a column vector of matrix A and a row vector of matrix B.
"""

from myhdl import *
from src.hdl.components.pe import pe


@block
def processing_array(
    clk,
    # Inputs
    i_a_vector,  # Input is a column vector of matrix A (rows*data_width bits)
    i_b_vector,  # Input is a row vector of matrix B (cols*data_width bits)
    i_data_valid,  # Data valid control signal
    i_read_en,  # Read enable signal (scalar, not vector)
    i_reset,  # Reset signal
    # Outputs
    o_c_matrix,  # Output matrix C (flattened, rows*cols*acc_width bits)
    o_saturate_detect,  # Overflow detection (for any PE)
    # Parameters
    rows=2,  # Number of rows in the array
    cols=2,  # Number of columns in the array
    data_width=8,  # Width of data inputs
    acc_width=16,  # Width of accumulators
):
    """
    Parallel Processing Array for matrix multiplication
    This implements a rows x cols array of processing elements (PEs)
    for performing matrix multiplication: C = A * B
    Each PE handles the multiplication of one element of the result matrix.
    """
    # Convert reset to proper signal type if needed
    if not isinstance(i_reset, ResetSignal):
        raise ValueError("Reset signal must be a ResetSignal")

    # Shadow signals for proper structural modeling
    a_slices = [Signal(intbv(0)[data_width:]) for _ in range(rows)]
    b_slices = [Signal(intbv(0)[data_width:]) for _ in range(cols)]

    # Connect shadow signals to input vectors - FIXED SLICING
    @always_comb
    def shadow_slices():
        for i in range(rows):
            # Correct bit slicing: [high:low]
            high_bit = (i + 1) * data_width
            low_bit = i * data_width
            a_slices[i].next = i_a_vector[high_bit:low_bit]

        for j in range(cols):
            high_bit = (j + 1) * data_width
            low_bit = j * data_width
            b_slices[j].next = i_b_vector[high_bit:low_bit]

    # PE outputs
    c_outputs = [Signal(intbv(0)[acc_width:]) for _ in range(rows * cols)]
    saturate_flags = [Signal(bool(0)) for _ in range(rows * cols)]

    # Instantiate the processing element array
    pe_instances = []
    for i in range(rows):
        for j in range(cols):
            # Calculate the index for the PE
            pe_idx = i * cols + j

            pe_instances.append(
                pe(
                    clk=clk,
                    i_a=a_slices[i],  # Use shadow signal
                    i_b=b_slices[j],  # Use shadow signal
                    i_data_valid=i_data_valid,
                    i_read_en=i_read_en,
                    i_reset=i_reset,
                    o_c=c_outputs[pe_idx],
                    o_saturate_detect=saturate_flags[pe_idx],
                    data_width=data_width,
                    acc_width=acc_width,
                )
            )

    # IMPROVED: Output collection logic using concat
    @always_comb
    def output_collection():
        # Method 1: Using MyHDL concat (most reliable)
        # Reverse the list because concat puts MSB first
        o_c_matrix.next = concat(*reversed(c_outputs))

        # Collect overflow flags (any PE overflow)
        overflow_detected = False
        for idx in range(rows * cols):
            if saturate_flags[idx]:
                overflow_detected = True
                break  # Early exit for efficiency

        o_saturate_detect.next = overflow_detected

    return instances()
