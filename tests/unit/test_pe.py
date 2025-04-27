import unittest
from myhdl import *
import sys
import os

# Import your module and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.hdl.components.pe import pe
from src.hdl.components.mac import mac
from tests.utils.hdl_test_utils import test_runner


class TestPeUnit(unittest.TestCase):

    def setUp(self):
        ### Test Values w/ Expected Results
        self.test_values = [
            (3, 4, 12),  # 3 * 4 = 12
            (5, 6, 30),  # 5 * 6 = 30
            (7, 8, 56),  # 7 * 8 = 56
            (9, 10, 90),  # 9 * 10 = 90
        ]
        ### Small for now
        self.data_width = 8
        self.acc_width = 16
        self.clk = Signal(bool(0))
        self.reset = ResetSignal(0, active=1, isasync=False)
        # Inputs
        self.i_a = Signal(intbv(0)[self.data_width : 0])
        self.i_b = Signal(intbv(0)[self.data_width : 0])
        self.i_data_valid = Signal(bool(0))
        self.i_read_en = Signal(bool(0))  # Read result signal
        # Outputs
        self.o_a = Signal(intbv(0)[self.data_width : 0])
        self.o_b = Signal(intbv(0)[self.data_width : 0])
        self.o_c = Signal(intbv(0)[self.acc_width : 0])  # Result
        self.o_saturate_detect = Signal(bool(0))

    # Dut creation
    def create_pe(self):
        return pe(
            clk=self.clk,
            i_a=self.i_a,
            i_b=self.i_b,
            i_data_valid=self.i_data_valid,
            i_read_en=self.i_read_en,
            i_reset=self.reset,
            o_a=self.o_a,
            o_b=self.o_b,
            o_c=self.o_c,
            o_saturate_detect=self.o_saturate_detect,
            data_width=self.data_width,
            acc_width=self.acc_width,
        )

    def testBasicFunction(self):
        """Test basic PE functionality with simple values."""

        @instance
        def test_sequence():
            self.i_a.next = self.test_values[0][0]
            self.i_b.next = self.test_values[0][1]
            self.i_data_valid.next = True
            self.i_read_en.next = False
            self.reset.next = False
            yield self.clk.posedge
            yield self.clk.negedge
            self.i_data_valid.next = False
            # Run the simulation for a few clock cycles
            for _ in range(5):
                yield self.clk.posedge

            self.i_read_en.next = True
            yield self.clk.posedge
            self.assertEqual(
                self.o_a,
                self.test_values[0][0],
                f"Expected {self.test_values[0][0]}, got {self.o_a}",
            )
            self.assertEqual(
                self.o_b,
                self.test_values[0][1],
                f"Expected {self.test_values[0][1]}, got {self.o_b}",
            )
            self.assertEqual(
                self.o_c,
                self.test_values[0][2],
                f"Expected {self.test_values[0][2]}, got {self.o_c}",
            )
            yield delay(5)
            self.i_read_en.next = False

            self.i_a.next = self.test_values[1][0]
            self.i_b.next = self.test_values[1][1]
            self.i_data_valid.next = True

            for _ in range(5):
                yield self.clk.posedge

            self.i_data_valid.next = False
            self.assertEqual(
                self.o_a,
                self.test_values[1][0],
                f"Expected {self.test_values[1][0]}, got {self.o_a}",
            )
            self.assertEqual(
                self.o_b,
                self.test_values[1][1],
                f"Expected {self.test_values[1][1]}, got {self.o_b}",
            )

            expected_result = (self.test_values[1][2] * 5) + (
                self.test_values[0][0] * self.test_values[0][1]
            )
            self.i_read_en.next = True
            yield self.clk.posedge
            self.assertEqual(
                self.o_c, expected_result, f"Expected {expected_result}, got {self.o_c}"
            )

        test_runner(
            self.create_pe,
            lambda: test_sequence,
            clk=self.clk,
            period=10,
            dut_name="pe",
            vcd_output=True,
            duration=500,
        )


if __name__ == "__main__":
    unittest.main()
