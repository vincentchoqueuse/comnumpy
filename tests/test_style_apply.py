"""`style.apply` decorates an axis and does not redraw it.

The function exists so that a figure drawn with two matplotlib calls
gets the same look as one drawn by `plot_error_rate`. That is only true
if it stays strictly a decoration: the moment it starts changing a
scale, a colour or a set of points, a page that calls it stops showing
what it draws, which is the whole reason it replaced the wrappers.

So the properties tested here are as much about what it must *not* do.
"""
import unittest

import matplotlib
matplotlib.use("Agg")                                   # no display
import matplotlib.pyplot as plt                         # noqa: E402
import numpy as np                                      # noqa: E402

from comnumpy import style                              # noqa: E402
from comnumpy.exceptions import ComnumpyError           # noqa: E402


class TestApply(unittest.TestCase):

    def setUp(self):
        self.figure, self.ax = plt.subplots()

    def tearDown(self):
        plt.close(self.figure)

    def test_fills_the_labels_of_each_kind(self):
        for kind, expected in style.KINDS.items():
            with self.subTest(kind=kind):
                _, ax = plt.subplots()
                style.apply(ax, kind)
                self.assertEqual(ax.get_xlabel(), expected["xlabel"])
                self.assertEqual(ax.get_ylabel(), expected["ylabel"])
                plt.close(ax.figure)

    def test_never_overwrites_a_label_the_caller_set(self):
        self.ax.set_xlabel("$E_b/N_0$ [dB]")
        self.ax.set_ylabel("BER")
        style.apply(self.ax, "error_rate")
        self.assertEqual(self.ax.get_xlabel(), "$E_b/N_0$ [dB]")
        self.assertEqual(self.ax.get_ylabel(), "BER")

    def test_leaves_a_title_alone(self):
        """No kind has a default title: only the page knows the subject."""
        style.apply(self.ax, "iq")
        self.assertEqual(self.ax.get_title(), "")

    def test_does_not_change_the_scales(self):
        """The caller chose `plot` or `semilogy`; that choice is the point."""
        self.ax.plot([1, 2], [1, 2])
        style.apply(self.ax, "error_rate")
        self.assertEqual(self.ax.get_yscale(), "linear")

    def test_does_not_touch_the_data(self):
        x, y = np.array([0.0, 1.0, 2.0]), np.array([1e-1, 1e-2, 1e-3])
        self.ax.semilogy(x, y, "o")
        before = self.ax.lines[0].get_xydata().copy()
        style.apply(self.ax, "error_rate")
        np.testing.assert_array_equal(self.ax.lines[0].get_xydata(), before)
        self.assertEqual(len(self.ax.lines), 1)

    def test_grid_runs_on_both_decades_only_when_logarithmic(self):
        self.ax.semilogy([1, 2], [1e-1, 1e-3])
        style.apply(self.ax, "error_rate")
        self.assertTrue(self.ax.yaxis.get_gridlines())
        minor_on_log = [line.get_visible()
                        for line in self.ax.yaxis.get_minorticklines()]
        self.assertTrue(any(minor_on_log))

    def test_legend_appears_only_when_something_is_labelled(self):
        self.ax.plot([1, 2], [1, 2])                    # no label
        style.apply(self.ax, "time")
        self.assertIsNone(self.ax.get_legend())

        self.ax.plot([1, 2], [2, 3], label="second")
        style.apply(self.ax, "time")
        self.assertIsNotNone(self.ax.get_legend())

    def test_legend_can_be_forced_either_way(self):
        self.ax.plot([1, 2], [1, 2], label="curve")
        style.apply(self.ax, "time", legend=False)
        self.assertIsNone(self.ax.get_legend())
        style.apply(self.ax, "time", legend=True)
        self.assertIsNotNone(self.ax.get_legend())

    def test_iq_gets_an_equal_aspect_ratio(self):
        """A constellation read on unequal axes is another constellation."""
        self.ax.plot([0, 1], [0, 2])
        style.apply(self.ax, "iq")
        self.assertEqual(self.ax.get_aspect(), 1.0)

    def test_unknown_kind_names_the_ones_that_exist(self):
        with self.assertRaises(ComnumpyError) as raised:
            style.apply(self.ax, "waterfall")
        message = str(raised.exception)
        self.assertIn("waterfall", message)
        for kind in style.KINDS:
            self.assertIn(kind, message)

    def test_returns_the_same_axis(self):
        self.assertIs(style.apply(self.ax, "iq"), self.ax)


class TestErrorRateStillMasksZeros(unittest.TestCase):
    """The one behaviour `apply` cannot provide, and must not be lost.

    A rate of zero means no error was seen, i.e. the estimate ran out of
    samples. matplotlib's default on a log axis is ``nonpositive="clip"``,
    which sends the point far below the axis, so a joined curve dives off
    the bottom of the figure and comes back -- a spike where there should
    be a gap. `plot_error_rate` sets ``"mask"`` instead.
    """

    def test_a_zero_is_masked_not_clipped(self):
        from comnumpy.core.visualizers import plot_error_rate
        rates = np.array([1e-1, 1e-2, 0.0, 1e-3])
        ax = plot_error_rate(np.arange(4.0), {"detector": rates})
        transform = ax.lines[0].get_transform()
        drawn = transform.transform_path(ax.lines[0].get_path()).vertices[:, 1]
        self.assertTrue(np.isinf(drawn[2]),
                        "the zero was clipped rather than masked; a joined "
                        "curve will spike to the bottom of the figure")
        self.assertTrue(np.isfinite(drawn[[0, 1, 3]]).all())
        plt.close(ax.figure)

    def test_a_zero_survives_a_linear_ordinate(self):
        """There a zero is an ordinary value, not a missing measurement."""
        from comnumpy.core.visualizers import plot_error_rate
        values = np.array([0.0, 1.0, 2.0, 3.0])
        ax = plot_error_rate(np.arange(4.0), {"rate": values},
                             yscale="linear", ylabel="bit/symbol")
        self.assertEqual(len(ax.lines[0].get_xdata()), 4)
        plt.close(ax.figure)


if __name__ == "__main__":
    unittest.main()
