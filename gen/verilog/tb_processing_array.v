module tb_processing_array;

reg clk;
reg [15:0] i_a_vector;
reg [15:0] i_b_vector;
reg i_data_valid;
reg i_read_en;
reg i_reset;
wire [63:0] o_c_matrix;
wire o_saturate_detect;

initial begin
    $from_myhdl(
        clk,
        i_a_vector,
        i_b_vector,
        i_data_valid,
        i_read_en,
        i_reset
    );
    $to_myhdl(
        o_c_matrix,
        o_saturate_detect
    );
end

processing_array dut(
    clk,
    i_a_vector,
    i_b_vector,
    i_data_valid,
    i_read_en,
    i_reset,
    o_c_matrix,
    o_saturate_detect
);

endmodule
