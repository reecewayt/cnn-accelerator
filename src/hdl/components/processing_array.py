"""
Parallel processing array for matrix multiplication, similar to a systolic array.
But all inputs are available at the same time, and the outputs are collected in a single clock cycle.
"""

from myhdl import *
from src.hdl.components.pe import pe

"TODO: This isn't complete yet. It needs to be tested and verified."


@block
def processing_array(
    clk,
    # Inputs
    i_a_matrix,  # Input matrix A (flattened)
    i_b_matrix,  # Input matrix B (flattened)
    i_data_valid,  # Data valid signal
    i_read_en,  # Read enable signal
    i_reset,  # Reset signal
    # Outputs
    o_c_matrix,  # Output matrix C (flattened)
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

    # Convert reset to proper signal type
    reset = ResetSignal(i_reset, active=1, isasync=False)

    # Create 2D arrays of signals for PE inputs and outputs
    # Matrix A and B inputs for each PE
    a_inputs = [
        [Signal(intbv(0)[data_width:]) for _ in range(cols)] for _ in range(rows)
    ]
    b_inputs = [
        [Signal(intbv(0)[data_width:]) for _ in range(cols)] for _ in range(rows)
    ]

    # PE outputs
    c_outputs = [
        [Signal(intbv(0)[acc_width:]) for _ in range(cols)] for _ in range(rows)
    ]
    saturate_flags = [[Signal(bool(0)) for _ in range(cols)] for _ in range(rows)]

    # Unused pass-through signals (not needed in this parallel design)
    a_pass = [[Signal(intbv(0)[data_width:]) for _ in range(cols)] for _ in range(rows)]
    b_pass = [[Signal(intbv(0)[data_width:]) for _ in range(cols)] for _ in range(rows)]

    # Instantiate the processing element array
    pe_array = []
    for i in range(rows):
        row_pes = []
        for j in range(cols):
            # Instantiate a PE for each position in the array
            pe_inst = pe(
                clk=clk,
                i_a=a_inputs[i][j],
                i_b=b_inputs[i][j],
                i_data_valid=i_data_valid,
                i_read_en=i_read_en,
                i_reset=reset,
                o_a=a_pass[i][j],  # Pass-through (unused in parallel design)
                o_b=b_pass[i][j],  # Pass-through (unused in parallel design)
                o_c=c_outputs[i][j],
                o_saturate_detect=saturate_flags[i][j],
                data_width=data_width,
                acc_width=acc_width,
            )
            row_pes.append(pe_inst)
        pe_array.append(row_pes)

    # Input distribution and output collection logic
    @always_comb
    def input_distribution():
        # For a 2x2 array, distribute the inputs
        # Matrix A distribution (row-wise): each PE in a row gets the same row elements
        for i in range(rows):
            for j in range(cols):
                # For each position in the output matrix,
                # we need one element from row i of A and one element from column j of B
                a_inputs[i][j].next = i_a_matrix[i * cols + j]
                b_inputs[i][j].next = i_b_matrix[i * rows + j]

    @always_comb
    def output_collection():
        # Collect output matrix elements
        for i in range(rows):
            for j in range(cols):
                o_c_matrix[i * cols + j].next = c_outputs[i][j]

        # Collect overflow flags (any PE overflow)
        o_saturate_detect.next = False
        for i in range(rows):
            for j in range(cols):
                if saturate_flags[i][j]:
                    o_saturate_detect.next = True

    return instances()
