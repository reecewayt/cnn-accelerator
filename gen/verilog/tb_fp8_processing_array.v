module tb_fp8_processing_array;

reg clk;
reg [15:0] i_a_vector;
reg [15:0] i_b_vector;
reg i_data_valid;
reg i_read_en;
reg i_reset;
reg i_clear_acc;
wire [31:0] o_c_matrix;
wire o_mac_done;
wire o_ready_for_new;

initial begin
    $from_myhdl(
        clk,
        i_a_vector,
        i_b_vector,
        i_data_valid,
        i_read_en,
        i_reset,
        i_clear_acc
    );
    $to_myhdl(
        o_c_matrix,
        o_mac_done,
        o_ready_for_new
    );
end

fp8_processing_array dut(
    clk,
    i_a_vector,
    i_b_vector,
    i_data_valid,
    i_read_en,
    i_reset,
    i_clear_acc,
    o_c_matrix,
    o_mac_done,
    o_ready_for_new
);

endmodule
