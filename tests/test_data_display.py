"""One result dictionary, two renderings that cannot disagree.

`print_data` and `plot_data` exist so that a table and a figure are two
views of the *same* object rather than two restatements of it. That
only holds if they read the structure through one checker, so the tests
below pin the contract (`unpack`) as much as the two renderers.

The formatting rules are pinned too, because a table is read by eye: a
column whose decimal points wander, or a `nan` printed as a number, is
a table that says something the run did not.
"""
import unittest

import matplotlib
matplotlib.use("Agg")                                   # no display
import matplotlib.pyplot as plt                         # noqa: E402
import numpy as np                                      # noqa: E402

from comnumpy.data import (format_data, plot_data,      # noqa: E402
                           print_data, unpack)
from comnumpy.exceptions import ComnumpyError, ShapeError   # noqa: E402


class TestUnpack(unittest.TestCase):

    def test_returns_float_arrays(self):
        x, curves = unpack({"x": [1, 2, 3], "curves": {"a": (4, 5, 6)}})
        self.assertEqual(x.dtype, np.float64)
        self.assertEqual(curves["a"].dtype, np.float64)
        self.assertEqual(list(curves), ["a"])

    def test_a_missing_key_says_what_the_structure_is(self):
        for data in ({"curves": {}}, {"x": [1]}):
            with self.subTest(data=data):
                with self.assertRaises(ComnumpyError) as raised:
                    unpack(data)
                self.assertIn("curves", str(raised.exception))

    def test_curves_must_be_a_mapping(self):
        """A bare list of series loses the names both renderings need."""
        with self.assertRaises(ComnumpyError):
            unpack({"x": [1, 2], "curves": [[3, 4]]})

    def test_a_short_series_is_a_shape_error_naming_the_curve(self):
        with self.assertRaises(ShapeError) as raised:
            unpack({"x": [1, 2, 3], "curves": {"ZF": [1, 2]}})
        message = str(raised.exception)
        self.assertIn("'ZF'", message)
        self.assertIn("3", message)

    def test_curve_order_is_the_insertion_order(self):
        """Columns and legend entries follow the order they were written."""
        _, curves = unpack({"x": [0], "curves": {"b": [1], "a": [2]}})
        self.assertEqual(list(curves), ["b", "a"])


class TestFormatData(unittest.TestCase):

    def table(self, data, **kwargs):
        return format_data(data, **kwargs).split("\n")

    def test_columns_are_aligned_and_as_wide_as_their_header(self):
        lines = self.table({"x": [1, 2],
                            "curves": {"a very long curve name": [1.5, 2.5]}},
                           xlabel="N")
        widths = {len(line) for line in lines}
        self.assertEqual(len(widths), 1, f"ragged table: {lines}")
        self.assertIn("a very long curve name", lines[0])

    def test_an_integral_column_keeps_no_decimals(self):
        lines = self.table({"x": [1000, 2000], "curves": {"n": [3, 4]}})
        self.assertIn("1000", lines[2])
        self.assertNotIn("1000.", lines[2])

    def test_a_column_spanning_decades_goes_scientific(self):
        lines = self.table({"x": [0, 1], "curves": {"SER": [7.4e-1, 1.2e-6]}})
        self.assertIn("7.400e-01", lines[2])
        self.assertIn("1.200e-06", lines[3])

    def test_a_narrow_column_stays_in_fixed_point(self):
        lines = self.table({"x": [0, 1], "curves": {"ms": [12.43, 98.71]}})
        self.assertIn("12.4", lines[2])
        self.assertIn("98.7", lines[3])

    def test_decimals_are_calibrated_on_the_smallest_entry(self):
        """Calibrating on the largest destroys exactly the small values.

        A column of 8.5, 42.1, 282.3, 2374.0 needs four significant
        digits on 2374 -- which is zero decimals, and prints the first
        entry as "8".
        """
        lines = self.table({"x": [0, 1, 2, 3],
                            "curves": {"ms": [8.5, 42.1, 282.3, 2374.0]}})
        self.assertIn("8.5", lines[2])
        self.assertIn("2374.0", lines[5])

    def test_no_decimal_the_values_do_not_carry(self):
        """A zero the measurement never had is a digit it did not measure."""
        lines = self.table({"x": [0, 1], "curves": {"dBm": [-6.0, -4.5]}})
        self.assertIn("-6.0", lines[2])
        self.assertNotIn("-6.00", lines[2])

    def test_transpose_puts_the_curves_in_rows(self):
        """The right way round when the names are the long part."""
        data = {"x": [-6.0, -4.5],
                "curves": {"dispersion compensation": [15.8, 17.1],
                           "DBP, 50 steps/span": [15.9, 17.4]}}
        wide = format_data(data, xlabel="launch power [dBm]")
        tall = format_data(data, xlabel="launch power [dBm]", transpose=True)
        self.assertLess(max(len(line) for line in tall.split("\n")),
                        max(len(line) for line in wide.split("\n")))
        self.assertTrue(tall.split("\n")[2].startswith(
            "dispersion compensation"))
        self.assertIn("15.8", tall.split("\n")[2])

    def test_transpose_shares_one_format_across_the_grid(self):
        """A row is read across, so a row of mixed formats is unreadable."""
        data = {"x": [0, 1], "curves": {"a": [1.5, 2.25], "b": [30.0, 40.0]}}
        rows = format_data(data, transpose=True).split("\n")[2:]
        decimals = {len(cell.split(".")[-1])
                    for row in rows for cell in row.split()[1:]}
        self.assertEqual(len(decimals), 1, rows)

    def test_the_format_is_chosen_per_column_not_per_cell(self):
        """A column read down must have its decimal points in one place."""
        data = {"x": [0, 1, 2], "curves": {"ms": [3.21, 12.4, 98.7]}}
        rows = self.table(data)[2:]
        decimals = {len(row.split(".")[-1]) for row in rows}
        self.assertEqual(len(decimals), 1, rows)

    def test_a_missing_point_is_not_printed_as_a_number(self):
        lines = self.table({"x": [0, 1], "curves": {"BER": [1e-3, np.nan]}})
        self.assertNotIn("nan", lines[3])
        self.assertTrue(lines[3].rstrip().endswith("-"), lines[3])

    def test_the_ylabel_is_a_caption_not_a_column(self):
        lines = self.table({"x": [0], "curves": {"a": [1.0]}},
                           ylabel="Runtime (ms)")
        self.assertEqual(lines[0], "Runtime (ms)")
        self.assertIn("a", lines[1])

    def test_one_row_per_point(self):
        data = {"x": np.arange(7.0), "curves": {"a": np.arange(7.0)}}
        self.assertEqual(len(self.table(data)), 2 + 7)   # header, rule, rows

    def test_print_data_writes_what_format_data_returns(self):
        import contextlib
        import io
        data = {"x": [0, 1], "curves": {"a": [1.0, 2.0]}}
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print_data(data)
        self.assertEqual(buffer.getvalue().rstrip("\n"), format_data(data))


class TestPlotData(unittest.TestCase):

    def setUp(self):
        self.data = {"x": [0, 1, 2],
                     "curves": {"ZF": [0.1, 0.05, 0.01],
                                "ML": [0.08, 0.03, 0.004]}}

    def test_one_line_per_curve_labelled_with_its_name(self):
        ax = plot_data(self.data)
        self.assertEqual(len(ax.lines), 2)
        _, labels = ax.get_legend_handles_labels()
        self.assertEqual(labels, ["ZF", "ML"])
        plt.close(ax.figure)

    def test_the_table_and_the_figure_show_the_same_numbers(self):
        """The point of the structure: one object, two renderings."""
        ax = plot_data(self.data)
        for line, (name, values) in zip(ax.lines, self.data["curves"].items(),
                                        strict=True):
            np.testing.assert_allclose(line.get_ydata(), values)
            self.assertIn(name, format_data(self.data))
        plt.close(ax.figure)

    def test_scales_are_left_alone_unless_asked(self):
        ax = plot_data(self.data)
        self.assertEqual(ax.get_yscale(), "linear")
        plt.close(ax.figure)
        ax = plot_data(self.data, yscale="log")
        self.assertEqual(ax.get_yscale(), "log")
        plt.close(ax.figure)

    def test_a_zero_on_a_log_ordinate_is_masked_not_clipped(self):
        data = {"x": [0, 1, 2], "curves": {"BER": [1e-1, 1e-3, 0.0]}}
        ax = plot_data(data, yscale="log")
        drawn = ax.lines[0].get_transform().transform_path(
            ax.lines[0].get_path()).vertices[:, 1]
        self.assertTrue(np.isinf(drawn[2]))
        plt.close(ax.figure)

    def test_it_draws_on_the_axis_it_is_given(self):
        _, ax = plt.subplots()
        self.assertIs(plot_data(self.data, ax=ax), ax)
        plt.close(ax.figure)

    def test_a_kind_hands_the_decoration_to_style_apply(self):
        ax = plot_data(self.data, kind="error_rate")
        self.assertEqual(ax.get_xlabel(), "SNR [dB]")
        self.assertEqual(ax.get_ylabel(), "error rate")
        plt.close(ax.figure)

    def test_explicit_labels_win_over_the_kind(self):
        ax = plot_data(self.data, kind="error_rate", ylabel="BER")
        self.assertEqual(ax.get_ylabel(), "BER")
        plt.close(ax.figure)

    def test_plot_kwargs_reach_every_curve(self):
        ax = plot_data(self.data, marker="s", linestyle="--")
        for line in ax.lines:
            self.assertEqual(line.get_marker(), "s")
            self.assertEqual(line.get_linestyle(), "--")
        plt.close(ax.figure)

    def test_a_bad_structure_is_refused_before_anything_is_drawn(self):
        with self.assertRaises(ShapeError):
            plot_data({"x": [1, 2, 3], "curves": {"a": [1, 2]}})


class TestMonteCarloShape(unittest.TestCase):
    """A sweep already returns `curves`; that is the reason for the format."""

    def test_a_sweep_result_is_a_curves_dictionary(self):
        from comnumpy import monte_carlo
        from comnumpy.core import Sequential
        from comnumpy.core.channels import AWGN
        from comnumpy.core.generators import SymbolGenerator
        from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
        from comnumpy.core.metrics import compute_ser
        from comnumpy.core.utils import Constellation

        constellation = Constellation("QAM", 4)
        chain = Sequential([
            SymbolGenerator(constellation.order, name="tx"),
            SymbolMapper(constellation),
            AWGN(snr_dB=0, name="awgn"),
            SymbolDemapper(constellation),
        ], taps=["tx"])
        snr_dB = np.arange(0, 9, 4)
        curves = monte_carlo(chain, "awgn.snr_dB", snr_dB,
                             {"ser": compute_ser}, 2000,
                             reference="tx", seed=1)
        x, unpacked = unpack({"x": snr_dB, "curves": curves})
        self.assertEqual(list(unpacked), ["ser"])
        self.assertEqual(unpacked["ser"].shape, x.shape)


if __name__ == "__main__":
    unittest.main()
