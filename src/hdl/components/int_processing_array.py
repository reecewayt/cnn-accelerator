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
    o_ready_for_data,
    o_overflow_detected,
):
    DATA_WIDTH = 8
    ACC_WIDTH = 32
    ARRAY_SIZE = 3
    NUM_PES = 9

    # Shadow signals to extract 8-bit elements
    a_elements = [Signal(intbv(0, min=-128, max=128)) for _ in range(ARRAY_SIZE)]
    b_elements = [Signal(intbv(0, min=-128, max=128)) for _ in range(ARRAY_SIZE)]

    @always_comb
    def vector_decomposition():
        for i in range(ARRAY_SIZE):
            a_elements[i].next = i_a_vector[(i + 1) * 8 : i * 8]
            b_elements[i].next = i_b_vector[(i + 1) * 8 : i * 8]

    # 3x3 matrix of output signals
    pe_results = [
        [Signal(intbv(0, min=-(2**31), max=2**31)) for _ in range(ARRAY_SIZE)]
        for _ in range(ARRAY_SIZE)
    ]
    pe_overflows = [
        [Signal(bool(0)) for _ in range(ARRAY_SIZE)] for _ in range(ARRAY_SIZE)
    ]
    pe_dones = [[Signal(bool(0)) for _ in range(ARRAY_SIZE)] for _ in range(ARRAY_SIZE)]

    # Internal control/state signals
    internal_enable = Signal(bool(0))
    result_reg = Signal(intbv(0)[NUM_PES * ACC_WIDTH :])
    t_State = enum("IDLE", "COMPUTING", "DONE")
    state = Signal(t_State.IDLE)

    # Aggregated flags
    @always_comb
    def aggregate_flags():
        done_sum = True
        overflow_sum = False
        for i in range(ARRAY_SIZE):
            for j in range(ARRAY_SIZE):
                done_sum = done_sum and pe_dones[i][j]
                overflow_sum = overflow_sum or pe_overflows[i][j]
        o_computation_done.next = state == t_State.DONE
        o_overflow_detected.next = overflow_sum
        o_ready_for_data.next = state == t_State.IDLE

    # Instantiate processing elements
    pe_instances = []
    for i in range(ARRAY_SIZE):
        for j in range(ARRAY_SIZE):
            pe = processing_element(
                clk=clk,
                i_reset=i_reset,
                i_a=a_elements[i],
                i_b=b_elements[j],
                i_enable=internal_enable,
                i_clear=i_clear_acc,
                o_result=pe_results[i][j],
                o_overflow=pe_overflows[i][j],
                o_done=pe_dones[i][j],
                data_width=DATA_WIDTH,
                acc_width=ACC_WIDTH,
            )
            pe_instances.append(pe)

    # State machine
    @always_seq(clk.posedge, reset=i_reset)
    def fsm():
        if state == t_State.IDLE:
            if i_data_valid:
                internal_enable.next = True
                state.next = t_State.COMPUTING
        elif state == t_State.COMPUTING:
            internal_enable.next = False
            if all(
                pe_dones[i][j] for i in range(ARRAY_SIZE) for j in range(ARRAY_SIZE)
            ):
                state.next = t_State.DONE
        elif state == t_State.DONE:
            if not i_data_valid:
                state.next = t_State.IDLE

    # Pack output result
    @always_seq(clk.posedge, reset=i_reset)
    def pack_output():
        if i_read_enable and state == t_State.DONE:
            for i in range(ARRAY_SIZE):
                for j in range(ARRAY_SIZE):
                    idx = i * ARRAY_SIZE + j
                    result_reg[(idx + 1) * ACC_WIDTH : idx * ACC_WIDTH] = pe_results[i][
                        j
                    ]
        o_result_matrix.next = result_reg

    return instances()
