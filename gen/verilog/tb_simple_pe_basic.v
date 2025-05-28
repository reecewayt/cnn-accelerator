module tb_simple_pe_basic;

reg clk;
reg i_reset;
reg [7:0] i_a;
reg [7:0] i_b;
reg i_enable;
reg i_clear;
wire [15:0] o_result;
wire o_overflow;

initial begin
    $from_myhdl(
        clk,
        i_reset,
        i_a,
        i_b,
        i_enable,
        i_clear
    );
    $to_myhdl(
        o_result,
        o_overflow
    );
end

simple_pe_basic dut(
    clk,
    i_reset,
    i_a,
    i_b,
    i_enable,
    i_clear,
    o_result,
    o_overflow
);

endmodule
