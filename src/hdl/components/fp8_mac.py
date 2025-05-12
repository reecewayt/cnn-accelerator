from myhdl import *
import sys
import os

from src.hdl.components.fp8_e4m3_mult import fp8_e4m3_multiply
from src.hdl.components.fp8_e4m3_add import fp8_e4m3_add

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.utils.fp_defs import E4M3Format


@block
def fp8_e4m3_mac(
    clk,
    rst,
    input_a,
    input_b,
    mac_start,
    clear_acc,
    read_enable,
    output_result,
    mac_done,
    ready_for_new,
):
    """
    Pipelined E4M3 floating-point MAC unit
    """
    WIDTH = E4M3Format.WIDTH  # 8

    # Multiplier interface
    mult_a = Signal(intbv(0)[WIDTH:])
    mult_b = Signal(intbv(0)[WIDTH:])
    mult_result = Signal(intbv(0)[WIDTH:])
    mult_start = Signal(bool(0))
    mult_done = Signal(bool(0))

    # Adder interface
    add_a = Signal(intbv(0)[WIDTH:])
    add_b = Signal(intbv(0)[WIDTH:])
    add_result = Signal(intbv(0)[WIDTH:])
    add_start = Signal(bool(0))
    add_done = Signal(bool(0))

    # Pipeline registers
    accumulator = Signal(intbv(0)[WIDTH:])
    mult_result_reg = Signal(intbv(0)[WIDTH:])
    mult_pending = Signal(bool(0))
    add_pending = Signal(bool(0))

    # Output register - only updates when read_enable is active
    output_reg = Signal(intbv(0)[WIDTH:])

    # State machines for multiply and accumulate pipelines
    t_MultState = enum("IDLE", "MULTIPLY", "WAIT_DONE")
    t_AccState = enum("IDLE", "ADD", "WAIT_ADD", "UPDATE")

    mult_state = Signal(t_MultState.IDLE)
    acc_state = Signal(t_AccState.IDLE)

    # Status signals
    s_mac_done = Signal(bool(0))
    s_ready_for_new = Signal(bool(0))

    # Instantiate multiplier
    multiplier = fp8_e4m3_multiply(
        input_a=mult_a,
        input_b=mult_b,
        output_z=mult_result,
        start=mult_start,
        done=mult_done,
        clk=clk,
        rst=rst,
    )

    # Instantiate adder
    adder = fp8_e4m3_add(
        input_a=add_a,
        input_b=add_b,
        output_z=add_result,
        start=add_start,
        done=add_done,
        clk=clk,
        rst=rst,
    )

    # Multiply pipeline state machine
    @always_seq(clk.posedge, reset=rst)
    def multiply_pipeline():
        if rst:
            mult_state.next = t_MultState.IDLE
        else:
            if mult_state == t_MultState.IDLE:
                if mac_start:
                    mult_a.next = input_a
                    mult_b.next = input_b
                    mult_state.next = t_MultState.MULTIPLY

            elif mult_state == t_MultState.MULTIPLY:
                mult_start.next = 1
                mult_state.next = t_MultState.WAIT_DONE

            elif mult_state == t_MultState.WAIT_DONE:
                mult_start.next = 0
                if mult_done:
                    mult_result_reg.next = mult_result
                    mult_state.next = t_MultState.IDLE

    # Accumulate pipeline state machine
    @always_seq(clk.posedge, reset=rst)
    def accumulate_pipeline():
        if rst:
            acc_state.next = t_AccState.IDLE
            accumulator.next = 0
            add_pending.next = 0
            s_mac_done.next = 0
        else:
            s_mac_done.next = 0  # Default to not done

            if acc_state == t_AccState.IDLE:
                if clear_acc:
                    # Clear accumulator
                    accumulator.next = 0
                    add_pending.next = 0
                elif mult_pending and not add_pending:
                    # Start accumulation
                    add_a.next = mult_result_reg
                    add_b.next = accumulator
                    add_pending.next = 1
                    acc_state.next = t_AccState.ADD

            elif acc_state == t_AccState.ADD:
                add_start.next = 1
                acc_state.next = t_AccState.WAIT_ADD

            elif acc_state == t_AccState.WAIT_ADD:
                add_start.next = 0
                if add_done:
                    acc_state.next = t_AccState.UPDATE

            elif acc_state == t_AccState.UPDATE:
                accumulator.next = add_result
                add_pending.next = 0
                s_mac_done.next = 1  # Signal that this MAC operation completed
                acc_state.next = t_AccState.IDLE

    # Separate process for mult_pending to avoid multiple drivers
    @always_seq(clk.posedge, reset=rst)
    def mult_pending_control():
        if rst:
            mult_pending.next = 0
        else:
            if mult_state == t_MultState.WAIT_DONE and mult_done:
                # Set when multiplication completes
                mult_pending.next = 1
            elif acc_state == t_AccState.UPDATE:
                # Clear when accumulation completes
                mult_pending.next = 0
            elif clear_acc:
                # Also clear on accumulator clear
                mult_pending.next = 0

    # Output register control
    @always_seq(clk.posedge, reset=rst)
    def output_control():
        if rst:
            output_reg.next = 0
        else:
            if read_enable:
                # Update output register with current accumulator value
                output_reg.next = accumulator

    @always_comb
    def output_logic():
        # Output is the registered value
        output_result.next = output_reg
        mac_done.next = s_mac_done

        # Ready for new input when:
        # 1. Multiplier is idle AND
        # 2. No pending multiplication result waiting to be accumulated
        s_ready_for_new.next = (mult_state == t_MultState.IDLE) and not mult_pending
        ready_for_new.next = s_ready_for_new

    return instances()
