"""
Floating-point processing array with 2x2 fixed size
"""

from myhdl import *
from src.hdl.components.fp8_pe import fp8_pe


@block
def fp8_processing_array(
    clk,
    # Inputs
    i_a_vector,  # Input is a column vector of matrix A (2*8 bits)
    i_b_vector,  # Input is a row vector of matrix B (2*8 bits)
    i_data_valid,  # Data valid control signal
    i_read_en,  # Read enable signal (scalar, not vector)
    i_reset,  # Reset signal
    i_clear_acc,  # Clear accumulator signal
    # Outputs
    o_c_matrix,  # Output matrix C (flattened, 4*8 bits)
    o_mac_done,  # Matrix operation done signal (all PEs done)
    o_ready_for_new,  # Ready for new input signal
):
    """
    Floating Point Parallel Processing Array (2x2) for matrix multiplication
    This implements a 2x2 array of floating point processing elements (FP8_PEs)
    for performing matrix multiplication: C = A * B
    Each PE handles the multiplication of one element of the result matrix.
    """
    # Constants for this fixed 2x2 implementation
    rows = 2
    cols = 2
    data_width = 8

    # Convert reset to proper signal type if needed
    if not isinstance(i_reset, ResetSignal):
        raise ValueError("Reset signal must be a ResetSignal")

    # Shadow signals for each matrix element
    a_slices = [Signal(intbv(0)[data_width:]) for _ in range(rows)]
    b_slices = [Signal(intbv(0)[data_width:]) for _ in range(cols)]

    # Connect shadow signals to input vectors with fixed indices
    @always_comb
    def shadow_slices():
        # Extract from matrix A - with fixed bit slices
        a_slices[0].next = i_a_vector[7:0]  # A[0,0] (first row)
        a_slices[1].next = i_a_vector[15:8]  # A[1,0] (second row)

        # Extract from matrix B - with fixed bit slices
        b_slices[0].next = i_b_vector[7:0]  # B[0,0] (first column)
        b_slices[1].next = i_b_vector[15:8]  # B[0,1] (second column)

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

    # Instantiate the 2x2 processing element array
    pe_instances = []

    # PE at position [0,0]
    pe_instances.append(
        fp8_pe(
            clk=clk,
            i_a=a_slices[0],
            i_b=b_slices[0],
            i_data_valid=i_data_valid,
            i_read_en=i_read_en,
            i_reset=i_reset,
            i_clear_acc=i_clear_acc,
            o_c=c_outputs[0],
            o_mac_done=mac_done_signals[0],
            o_ready_for_new=ready_signals[0],
            data_width=data_width,
        )
    )

    # PE at position [0,1]
    pe_instances.append(
        fp8_pe(
            clk=clk,
            i_a=a_slices[0],
            i_b=b_slices[1],
            i_data_valid=i_data_valid,
            i_read_en=i_read_en,
            i_reset=i_reset,
            i_clear_acc=i_clear_acc,
            o_c=c_outputs[1],
            o_mac_done=mac_done_signals[1],
            o_ready_for_new=ready_signals[1],
            data_width=data_width,
        )
    )

    # PE at position [1,0]
    pe_instances.append(
        fp8_pe(
            clk=clk,
            i_a=a_slices[1],
            i_b=b_slices[0],
            i_data_valid=i_data_valid,
            i_read_en=i_read_en,
            i_reset=i_reset,
            i_clear_acc=i_clear_acc,
            o_c=c_outputs[2],
            o_mac_done=mac_done_signals[2],
            o_ready_for_new=ready_signals[2],
            data_width=data_width,
        )
    )

    # PE at position [1,1]
    pe_instances.append(
        fp8_pe(
            clk=clk,
            i_a=a_slices[1],
            i_b=b_slices[1],
            i_data_valid=i_data_valid,
            i_read_en=i_read_en,
            i_reset=i_reset,
            i_clear_acc=i_clear_acc,
            o_c=c_outputs[3],
            o_mac_done=mac_done_signals[3],
            o_ready_for_new=ready_signals[3],
            data_width=data_width,
        )
    )

    # Sequential logic to latch done signals and update output register
    @always_seq(clk.posedge, reset=i_reset)
    def pe_done_latch_logic():
        if i_reset:
            # Reset all latches and the output register
            pe_done_latches[0].next = 0
            pe_done_latches[1].next = 0
            pe_done_latches[2].next = 0
            pe_done_latches[3].next = 0
            all_pes_done.next = 0
            output_matrix_reg.next = 0
        elif i_data_valid:
            # New operation starting, clear the done latches
            pe_done_latches[0].next = 0
            pe_done_latches[1].next = 0
            pe_done_latches[2].next = 0
            pe_done_latches[3].next = 0
            all_pes_done.next = 0
        else:
            # For each PE, if it signals done, latch that information
            all_done = True

            # Check PE [0,0]
            if mac_done_signals[0]:
                pe_done_latches[0].next = 1
            all_done = all_done and pe_done_latches[0]

            # Check PE [0,1]
            if mac_done_signals[1]:
                pe_done_latches[1].next = 1
            all_done = all_done and pe_done_latches[1]

            # Check PE [1,0]
            if mac_done_signals[2]:
                pe_done_latches[2].next = 1
            all_done = all_done and pe_done_latches[2]

            # Check PE [1,1]
            if mac_done_signals[3]:
                pe_done_latches[3].next = 1
            all_done = all_done and pe_done_latches[3]

            # If all PEs are latched as done, set the all_pes_done signal
            all_pes_done.next = all_done

            # Update output register if read is enabled
            if i_read_en:
                temp = intbv(0)[rows * cols * data_width : 0]

                # Position [0,0]
                temp[7:0] = c_outputs[0]

                # Position [0,1]
                temp[15:8] = c_outputs[1]

                # Position [1,0]
                temp[23:16] = c_outputs[2]

                # Position [1,1]
                temp[31:24] = c_outputs[3]

                # Update the output register
                output_matrix_reg.next = temp

    # Connect output register to output port
    @always_comb
    def output_connection():
        o_mac_done.next = all_pes_done
        if i_read_en:
            o_c_matrix.next = output_matrix_reg
        else:
            o_c_matrix.next = output_matrix_reg

    # Generate 'AND' reduction for ready signals in a synthesizable way
    @always_comb
    def ready_logic():
        o_ready_for_new.next = (
            ready_signals[0]
            and ready_signals[1]
            and ready_signals[2]
            and ready_signals[3]
        )

    # Return all processes and instances
    return instances()
