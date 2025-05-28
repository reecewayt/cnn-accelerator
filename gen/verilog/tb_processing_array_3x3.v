module tb_processing_array_3x3;

reg clk;
reg i_reset;
reg [23:0] i_a_vector;
reg [23:0] i_b_vector;
reg i_data_valid;
reg i_read_enable;
reg i_clear_acc;
wire [287:0] o_result_matrix;
wire o_computation_done;
wire o_overflow_detected;

initial begin
    $from_myhdl(
        clk,
        i_reset,
        i_a_vector,
        i_b_vector,
        i_data_valid,
        i_read_enable,
        i_clear_acc
    );
    $to_myhdl(
        o_result_matrix,
        o_computation_done,
        o_overflow_detected
    );
end

processing_array_3x3 dut(
    clk,
    i_reset,
    i_a_vector,
    i_b_vector,
    i_data_valid,
    i_read_enable,
    i_clear_acc,
    o_result_matrix,
    o_computation_done,
    o_overflow_detected
);

endmodule
