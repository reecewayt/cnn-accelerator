module tb_mac;

reg clk;
reg reset;
reg [7:0] a;
reg [7:0] b;
reg clear;
wire [16:0] result;

initial begin
    $from_myhdl(
        clk,
        reset,
        a,
        b,
        clear
    );
    $to_myhdl(
        result
    );
end

mac dut(
    clk,
    reset,
    a,
    b,
    clear,
    result
);

endmodule
