module tb_fp8_pe;

reg clk;
reg [7:0] i_a;
reg [7:0] i_b;
reg i_data_valid;
reg i_read_en;
reg i_reset;
reg i_clear_acc;
wire [7:0] o_c;
wire o_mac_done;
wire o_ready_for_new;

initial begin
    $from_myhdl(
        clk,
        i_a,
        i_b,
        i_data_valid,
        i_read_en,
        i_reset,
        i_clear_acc
    );
    $to_myhdl(
        o_c,
        o_mac_done,
        o_ready_for_new
    );
end

fp8_pe dut(
    clk,
    i_a,
    i_b,
    i_data_valid,
    i_read_en,
    i_reset,
    i_clear_acc,
    o_c,
    o_mac_done,
    o_ready_for_new
);

endmodule
