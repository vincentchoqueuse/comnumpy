"""Observation surface: plotting functions and signal_report.

Since the Recorder/Logger/Scope blocks are gone, observation is made of
plain functions applied to arrays extracted with ``Sequential(taps=...)``.
These tests pin the contracts those functions must honour: return the
axis, never call ``plt.show()``, never touch the input.
"""
import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from comnumpy.core.metrics import signal_report  # noqa: E402
from comnumpy.core.visualizers import (plot_iq, plot_kde,  # noqa: E402
                                       plot_spectrum, plot_time, plot_welch)
from comnumpy.ofdm.visualizers import plot_subcarrier_amplitude  # noqa: E402

PLOTTERS = (plot_time, plot_spectrum, plot_iq)


class TestPlottingContract(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.normal(size=256) + 1j * rng.normal(size=256)
        self.addCleanup(plt.close, "all")

    def test_returns_the_axis_it_drew_on(self):
        for plot in PLOTTERS:
            with self.subTest(plot=plot.__name__):
                _, ax = plt.subplots()
                self.assertIs(plot(self.x, ax=ax), ax)

    def test_creates_an_axis_when_none_given(self):
        for plot in PLOTTERS + (plot_welch, plot_kde):
            with self.subTest(plot=plot.__name__):
                ax = plot(self.x)
                self.assertIsInstance(ax, plt.Axes)

    def test_input_is_left_untouched(self):
        reference = self.x.copy()
        for plot in PLOTTERS + (plot_welch, plot_kde):
            plot(self.x)
        np.testing.assert_array_equal(self.x, reference)

    def test_multi_stream_input_is_overlaid(self):
        streams = np.stack([self.x, 2 * self.x])
        ax = plot_time(streams)
        self.assertEqual(len(ax.lines), 2)

    def test_rejects_more_than_two_dimensions(self):
        with self.assertRaises(ValueError):
            plot_iq(np.zeros((2, 3, 4)))

    def test_subcarrier_amplitude(self):
        X = np.ones((4, 8), dtype=complex)
        ax = plot_subcarrier_amplitude(X)
        self.assertIsInstance(ax, plt.Axes)
        with self.assertRaises(ValueError):
            plot_subcarrier_amplitude(X, reduction="median")
        with self.assertRaises(ValueError):
            plot_subcarrier_amplitude(np.ones(8))


class TestSignalReport(unittest.TestCase):

    def test_statistics_of_a_unit_power_signal(self):
        report = signal_report(np.array([1.0 + 0j, 0.0 + 1j]))
        self.assertEqual(report["avg_power"], 1.0)
        self.assertEqual(report["rms"], 1.0)
        self.assertEqual(report["energy"], 2.0)
        self.assertEqual(report["min"], 1.0)
        self.assertEqual(report["max"], 1.0)
        self.assertNotIn("papr", report)

    def test_papr_is_opt_in(self):
        report = signal_report(np.array([1.0, 2.0, 3.0, 4.0]),
                               compute_papr=True)
        self.assertAlmostEqual(report["papr"], 3.2906, places=3)

    def test_values_are_plain_floats(self):
        for value in signal_report(np.arange(10.0)).values():
            self.assertIsInstance(value, float)


if __name__ == "__main__":
    unittest.main()
