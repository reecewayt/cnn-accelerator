module tb_fp8_e4m3_mac;

reg clk;
reg rst;
reg [7:0] input_a;
reg [7:0] input_b;
reg mac_start;
reg clear_acc;
reg read_enable;
wire [7:0] output_result;
wire mac_done;
wire ready_for_new;

initial begin
    $from_myhdl(
        clk,
        rst,
        input_a,
        input_b,
        mac_start,
        clear_acc,
        read_enable
    );
    $to_myhdl(
        output_result,
        mac_done,
        ready_for_new
    );
end

fp8_e4m3_mac dut(
    clk,
    rst,
    input_a,
    input_b,
    mac_start,
    clear_acc,
    read_enable,
    output_result,
    mac_done,
    ready_for_new
);

endmodule
