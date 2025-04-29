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

    # We need shadow signals to do structural modeling as described
    # in the MyHDL documentation.
    a_slices = [Signal(intbv(0)[data_width:0]) for _ in range(rows)]
    b_slices = [Signal(intbv(0)[data_width:0]) for _ in range(cols)]

    # Connect shadow signal to input vectors
    @always_comb
    def shadow_slices():
        for i in range(rows):
            a_slices[i].next = i_a_vector[(i + 1) * data_width - 1 : i * data_width]
        for j in range(cols):
            b_slices[j].next = i_b_vector[(j + 1) * data_width - 1 : j * data_width]

    # PE outputs
    c_outputs = [Signal(intbv(0)[acc_width:0]) for _ in range(rows * cols)]
    saturate_flags = [Signal(bool(0)) for _ in range(rows * cols)]

    # Instantiate the processing element array
    pe_instances = []
    for i in range(rows):
        for j in range(cols):
            # Calculate the index for the PE
            pe_idx = i * cols + j

            # Use shadow signals for structural connections
            pe_inst = pe(
                clk=clk,
                i_a=a_slices[i],  # Use shadow signal instead of slice
                i_b=b_slices[j],  # Use shadow signal instead of slice
                i_data_valid=i_data_valid,
                i_read_en=i_read_en,
                i_reset=i_reset,
                o_c=c_outputs[pe_idx],
                o_saturate_detect=saturate_flags[pe_idx],
                data_width=data_width,
                acc_width=acc_width,
            )
            pe_instances.append(pe_inst)

    # Output collection logic

    # @always_comb
    # def output_collection():
    # Collect all PE outputs into the flattened output matrix
    # if i_read_en:
    #    o_c_matrix.next = c_outputs
    #    o_saturate_detect.next = saturate_flags

    # else:
    #    o_c_matrix.next = 0

    # return instances()
    return pe_instances, shadow_slices
