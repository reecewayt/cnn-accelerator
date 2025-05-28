"""
Refactored 3x3 Integer Processing Array for matrix multiplication
Following exact signal slicing pattern from fp8_processing_array
"""

from myhdl import *
from src.hdl.components.pe import processing_element


@block
def processing_array_3x3(
    clk,
    i_reset,
    i_a_vector,
    i_b_vector,
    i_data_valid,
    i_read_enable,
    i_clear_acc,
    o_result_matrix,
    o_computation_done,
    o_overflow_detected,
):
    """
    3x3 Processing Array for matrix multiplication using validated processing elements.

    This array performs matrix multiplication by computing dot products in parallel.
    Each PE accumulates partial results across multiple cycles.

    Parameters:
    - clk: Clock signal
    - i_reset: Reset signal (active high)
    - i_a_vector: Column vector from matrix A (24 bits = 3 x 8-bit elements)
    - i_b_vector: Row vector from matrix B (24 bits = 3 x 8-bit elements)
    - i_data_valid: Start computation when high
    - i_read_enable: Enable reading results
    - i_clear_acc: Clear all accumulators
    - o_result_matrix: Flattened 3x3 result matrix (288 bits = 9 x 32-bit elements)
    - o_computation_done: All PEs completed their MAC operations
    - o_overflow_detected: At least one PE detected overflow
    """

    # Constants
    DATA_WIDTH = 8
    ACC_WIDTH = 32
    ARRAY_SIZE = 3
    NUM_PES = 9

    # Validate reset signal type
    if not isinstance(i_reset, ResetSignal):
        raise ValueError("Reset signal must be a ResetSignal")

    # Shadow signals for each matrix element - following fp8_processing_array pattern
    a_slices = [Signal(intbv(0, min=-128, max=128)) for _ in range(ARRAY_SIZE)]
    b_slices = [Signal(intbv(0, min=-128, max=128)) for _ in range(ARRAY_SIZE)]

    t_State = enum("IDLE", "PROCESSING")
    state = Signal(t_State.IDLE)

    temp_result_matrix = Signal(
        intbv(0)[NUM_PES * ACC_WIDTH : 0]
    )  # Temporary result storage

    # Connect shadow signals to input vectors with fixed indices - exact pattern from fp8
    @always_comb
    def shadow_slices():
        # Extract from matrix A - with fixed bit slices
        a_slices[0].next = i_a_vector[7:0]  # A[0] (first element)
        a_slices[1].next = i_a_vector[15:8]  # A[1] (second element)
        a_slices[2].next = i_a_vector[23:16]  # A[2] (third element)

        # Extract from matrix B - with fixed bit slices
        b_slices[0].next = i_b_vector[7:0]  # B[0] (first element)
        b_slices[1].next = i_b_vector[15:8]  # B[1] (second element)
        b_slices[2].next = i_b_vector[23:16]  # B[2] (third element)

    # PE outputs - following fp8_processing_array pattern
    pe_results = [Signal(intbv(0, min=-(2**31), max=2**31)) for _ in range(NUM_PES)]
    pe_overflows = [Signal(bool(0)) for _ in range(NUM_PES)]
    pe_dones = [Signal(bool(0)) for _ in range(NUM_PES)]
    all_pes_done = Signal(bool(0))
    all_pes_done_latch = Signal(bool(0))

    # output_matrix_reg = Signal(intbv(0)[NUM_PES * ACC_WIDTH : 0])

    # Add latches for tracking which PEs are done - following fp8 pattern
    # pe_done_latches = [Signal(bool(0)) for _ in range(NUM_PES)]
    all_pes_done = Signal(bool(0))

    # Instantiate the 3x3 processing element array - explicit instantiation like fp8
    pe_instances = []

    # PE at position [0,0] - uses a_slices[0] and b_slices[0]
    pe_instances.append(
        processing_element(
            clk=clk,
            i_reset=i_reset,
            i_a=a_slices[0],
            i_b=b_slices[0],
            i_enable=i_data_valid,
            i_clear=i_clear_acc,
            o_result=pe_results[0],
            o_overflow=pe_overflows[0],
            o_done=pe_dones[0],
            data_width=DATA_WIDTH,
            acc_width=ACC_WIDTH,
        )
    )

    # PE at position [0,1] - uses a_slices[0] and b_slices[1]
    pe_instances.append(
        processing_element(
            clk=clk,
            i_reset=i_reset,
            i_a=a_slices[0],
            i_b=b_slices[1],
            i_enable=i_data_valid,
            i_clear=i_clear_acc,
            o_result=pe_results[1],
            o_overflow=pe_overflows[1],
            o_done=pe_dones[1],
            data_width=DATA_WIDTH,
            acc_width=ACC_WIDTH,
        )
    )

    # PE at position [0,2] - uses a_slices[0] and b_slices[2]
    pe_instances.append(
        processing_element(
            clk=clk,
            i_reset=i_reset,
            i_a=a_slices[0],
            i_b=b_slices[2],
            i_enable=i_data_valid,
            i_clear=i_clear_acc,
            o_result=pe_results[2],
            o_overflow=pe_overflows[2],
            o_done=pe_dones[2],
            data_width=DATA_WIDTH,
            acc_width=ACC_WIDTH,
        )
    )

    # PE at position [1,0] - uses a_slices[1] and b_slices[0]
    pe_instances.append(
        processing_element(
            clk=clk,
            i_reset=i_reset,
            i_a=a_slices[1],
            i_b=b_slices[0],
            i_enable=i_data_valid,
            i_clear=i_clear_acc,
            o_result=pe_results[3],
            o_overflow=pe_overflows[3],
            o_done=pe_dones[3],
            data_width=DATA_WIDTH,
            acc_width=ACC_WIDTH,
        )
    )

    # PE at position [1,1] - uses a_slices[1] and b_slices[1]
    pe_instances.append(
        processing_element(
            clk=clk,
            i_reset=i_reset,
            i_a=a_slices[1],
            i_b=b_slices[1],
            i_enable=i_data_valid,
            i_clear=i_clear_acc,
            o_result=pe_results[4],
            o_overflow=pe_overflows[4],
            o_done=pe_dones[4],
            data_width=DATA_WIDTH,
            acc_width=ACC_WIDTH,
        )
    )

    # PE at position [1,2] - uses a_slices[1] and b_slices[2]
    pe_instances.append(
        processing_element(
            clk=clk,
            i_reset=i_reset,
            i_a=a_slices[1],
            i_b=b_slices[2],
            i_enable=i_data_valid,
            i_clear=i_clear_acc,
            o_result=pe_results[5],
            o_overflow=pe_overflows[5],
            o_done=pe_dones[5],
            data_width=DATA_WIDTH,
            acc_width=ACC_WIDTH,
        )
    )

    # PE at position [2,0] - uses a_slices[2] and b_slices[0]
    pe_instances.append(
        processing_element(
            clk=clk,
            i_reset=i_reset,
            i_a=a_slices[2],
            i_b=b_slices[0],
            i_enable=i_data_valid,
            i_clear=i_clear_acc,
            o_result=pe_results[6],
            o_overflow=pe_overflows[6],
            o_done=pe_dones[6],
            data_width=DATA_WIDTH,
            acc_width=ACC_WIDTH,
        )
    )

    # PE at position [2,1] - uses a_slices[2] and b_slices[1]
    pe_instances.append(
        processing_element(
            clk=clk,
            i_reset=i_reset,
            i_a=a_slices[2],
            i_b=b_slices[1],
            i_enable=i_data_valid,
            i_clear=i_clear_acc,
            o_result=pe_results[7],
            o_overflow=pe_overflows[7],
            o_done=pe_dones[7],
            data_width=DATA_WIDTH,
            acc_width=ACC_WIDTH,
        )
    )

    # PE at position [2,2] - uses a_slices[2] and b_slices[2]
    pe_instances.append(
        processing_element(
            clk=clk,
            i_reset=i_reset,
            i_a=a_slices[2],
            i_b=b_slices[2],
            i_enable=i_data_valid,
            i_clear=i_clear_acc,
            o_result=pe_results[8],
            o_overflow=pe_overflows[8],
            o_done=pe_dones[8],
            data_width=DATA_WIDTH,
            acc_width=ACC_WIDTH,
        )
    )

    @always_seq(clk.posedge, reset=i_reset)
    def fsm_control_logic():
        if i_reset:
            state.next = t_State.IDLE
            o_computation_done.next = False
            o_overflow_detected.next = False
            all_pes_done.next = False
        else:
            if state == t_State.IDLE:
                all_pes_done.next = False
                o_computation_done.next = False
                if i_data_valid:
                    state.next = t_State.PROCESSING

            elif state == t_State.PROCESSING:
                # Check if all PEs are done
                if all_pes_done:
                    state.next = t_State.IDLE
                    o_computation_done.next = True
                else:
                    state.next = t_State.PROCESSING
                    o_computation_done.next = False

    @always_comb
    def pe_done_logic():
        # Update all_pes_done based on individual PE done signals
        if state == t_State.PROCESSING:
            all_pes_done.next = (
                pe_dones[0]
                and pe_dones[1]
                and pe_dones[2]
                and pe_dones[3]
                and pe_dones[4]
                and pe_dones[5]
                and pe_dones[6]
                and pe_dones[7]
                and pe_dones[8]
            )

    # Overflow detection - OR reduction following fp8 pattern
    @always_comb
    def overflow_logic():
        o_overflow_detected.next = (
            pe_overflows[0]
            or pe_overflows[1]
            or pe_overflows[2]
            or pe_overflows[3]
            or pe_overflows[4]
            or pe_overflows[5]
            or pe_overflows[6]
            or pe_overflows[7]
            or pe_overflows[8]
        )

    @always_seq(clk.posedge, reset=i_reset)
    def result_matrix_logic():
        if i_reset or i_clear_acc:
            temp_result_matrix.next = 0
        else:
            if all_pes_done:
                temp_result_matrix.next = ConcatSignal(*reversed(pe_results))
            else:
                temp_result_matrix.next = temp_result_matrix

    @always_comb
    def output_logic():
        if i_read_enable:
            o_result_matrix.next = temp_result_matrix

    # Return all processes and instances
    return instances()
