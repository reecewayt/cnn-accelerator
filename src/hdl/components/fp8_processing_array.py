"""
Improved floating-point processing array with latched done signals
"""

from myhdl import *
from src.hdl.components.fp8_pe import fp8_pe


@block
def fp8_processing_array(
    clk,
    # Inputs
    i_a_vector,  # Input is a column vector of matrix A (rows*data_width bits)
    i_b_vector,  # Input is a row vector of matrix B (cols*data_width bits)
    i_data_valid,  # Data valid control signal
    i_read_en,  # Read enable signal (scalar, not vector)
    i_reset,  # Reset signal
    i_clear_acc,  # Clear accumulator signal
    # Outputs
    o_c_matrix,  # Output matrix C (flattened, rows*cols*data_width bits)
    o_mac_done,  # Matrix operation done signal (all PEs done)
    o_ready_for_new,  # Ready for new input signal
    # Parameters
    rows=2,  # Number of rows in the array
    cols=2,  # Number of columns in the array
    data_width=8,  # Width of data inputs (E4M3 format: 8 bits)
):
    """
    Floating Point Parallel Processing Array for matrix multiplication
    This implements a rows x cols array of floating point processing elements (FP8_PEs)
    for performing matrix multiplication: C = A * B
    Each PE handles the multiplication of one element of the result matrix.
    """
    # Convert reset to proper signal type if needed
    if not isinstance(i_reset, ResetSignal):
        raise ValueError("Reset signal must be a ResetSignal")

    # We need shadow signals to do structural modeling
    a_slices = [Signal(intbv(0)[data_width:]) for _ in range(rows)]
    b_slices = [Signal(intbv(0)[data_width:]) for _ in range(cols)]

    # Connect shadow signal to input vectors
    @always_comb
    def shadow_slices():
        for i in range(rows):
            a_slices[i].next = i_a_vector[(i + 1) * data_width - 1 : i * data_width]
        for j in range(cols):
            b_slices[j].next = i_b_vector[(j + 1) * data_width - 1 : j * data_width]

    # PE outputs
    c_outputs = [Signal(intbv(0)[data_width:]) for _ in range(rows * cols)]
    mac_done_signals = [Signal(bool(0)) for _ in range(rows * cols)]
    ready_signals = [Signal(bool(0)) for _ in range(rows * cols)]

    # Create an output register to store the results
    output_matrix_reg = Signal(intbv(0)[rows * cols * data_width : 0])

    # Add latches for tracking which PEs are done
    pe_done_latches = [Signal(bool(0)) for _ in range(rows * cols)]

    # Signal to indicate all PEs are done
    all_pes_done = Signal(bool(0))

    # Instantiate the processing element array
    pe_instances = []
    for i in range(rows):
        for j in range(cols):
            # Calculate the index for the PE
            pe_idx = i * cols + j

            # Instantiate a floating point PE
            pe_instances.append(
                fp8_pe(
                    clk=clk,
                    i_a=a_slices[i],  # Use shadow signal
                    i_b=b_slices[j],  # Use shadow signal
                    i_data_valid=i_data_valid,
                    i_read_en=i_read_en,  # Pass the read enable signal to all PEs
                    i_reset=i_reset,
                    i_clear_acc=i_clear_acc,
                    o_c=c_outputs[pe_idx],
                    o_mac_done=mac_done_signals[pe_idx],
                    o_ready_for_new=ready_signals[pe_idx],
                    data_width=data_width,
                )
            )

    # Sequential logic to latch done signals and update output register
    @always_seq(clk.posedge, reset=i_reset)
    def pe_done_latch_logic():
        if i_reset:
            # Reset all latches and the output register
            for i in range(rows * cols):
                pe_done_latches[i].next = 0
            all_pes_done.next = 0
            output_matrix_reg.next = 0
        elif i_data_valid:
            # New operation starting, clear the done latches
            for i in range(rows * cols):
                pe_done_latches[i].next = 0
            all_pes_done.next = 0
        else:
            # For each PE, if it signals done, latch that information
            all_done = True
            for i in range(rows * cols):
                if mac_done_signals[i]:
                    pe_done_latches[i].next = 1
                all_done = all_done and pe_done_latches[i]

            # If all PEs are latched as done, set the all_pes_done signal
            all_pes_done.next = all_done

            if i_read_en:
                temp = intbv(0)[rows * cols * data_width : 0]

                # Collect all PE outputs into the flattened output matrix
                for idx in range(rows * cols):
                    # Calculate bit positions
                    high = (idx + 1) * data_width - 1
                    low = idx * data_width

                    # Copy each element
                    temp[high:low] = c_outputs[idx]

                # Update the output register
                output_matrix_reg.next = temp

    # Connect output register to output port
    @always_comb
    def output_connection():
        o_mac_done.next = all_pes_done  # Use latched done signal
        if i_read_en:
            o_c_matrix.next = output_matrix_reg

    # Generate 'AND' reduction for ready signals in a synthesizable way
    @always_comb
    def ready_logic():
        # Start with True and AND with each signal
        ready_temp = True
        for i in range(rows * cols):
            ready_temp = ready_temp and ready_signals[i]

        o_ready_for_new.next = ready_temp

    # Return all processes and instances
    return instances()
