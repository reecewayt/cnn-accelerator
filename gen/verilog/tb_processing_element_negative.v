module tb_processing_element_negative;

reg clk;
reg i_reset;
reg [7:0] i_a;
reg [7:0] i_b;
reg i_enable;
reg i_clear;
wire [31:0] o_result;
wire o_overflow;
wire o_done;

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
        o_overflow,
        o_done
    );
end

processing_element_negative dut(
    clk,
    i_reset,
    i_a,
    i_b,
    i_enable,
    i_clear,
    o_result,
    o_overflow,
    o_done
);

endmodule
