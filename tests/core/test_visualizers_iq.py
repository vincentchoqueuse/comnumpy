"""``plot_iq`` overlays and labels, added because ten examples drew them.

Every tutorial that shows a constellation was writing the same three
lines: scatter the real and imaginary parts, scatter the alphabet as
black crosses on top, and legend the two. A received cloud means little
without the points it is supposed to be near, so the overlay is not a
garnish -- it is what makes the picture a measurement. It belongs in the
plotting function, not in ten copies.

The tests below pin what the examples rely on: an overlay is drawn last
and in black crosses so it reads as the reference; a label replaces the
automatic per-stream naming so two signals can share one axis; and the
legend appears exactly when there is something to disambiguate.
"""
import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from comnumpy.core.utils import get_alphabet  # noqa: E402
from comnumpy.core.visualizers import plot_iq  # noqa: E402


class TestTheReferenceOverlay(unittest.TestCase):

    def setUp(self):
        self.alphabet = get_alphabet("QAM", 16)
        rng = np.random.default_rng(0)
        self.received = (self.alphabet[rng.integers(0, 16, 100)]
                         + 0.05 * rng.standard_normal(100))
        self.addCleanup(plt.close, "all")

    def test_without_a_reference_nothing_extra_is_drawn(self):
        ax = plot_iq(self.received)
        self.assertEqual(len(ax.lines), 1)
        self.assertIsNone(ax.get_legend())

    def test_the_reference_is_the_last_line_and_is_black_crosses(self):
        ax = plot_iq(self.received, reference=self.alphabet)
        self.assertEqual(len(ax.lines), 2)
        overlay = ax.lines[-1]
        self.assertEqual(overlay.get_color(), "k")
        self.assertEqual(overlay.get_marker(), "x")
        self.assertEqual(overlay.get_linestyle(), "None")

    def test_the_reference_carries_the_points_it_was_given(self):
        ax = plot_iq(self.received, reference=self.alphabet)
        overlay = ax.lines[-1]
        np.testing.assert_allclose(overlay.get_xdata(),
                                   np.real(self.alphabet))
        np.testing.assert_allclose(overlay.get_ydata(),
                                   np.imag(self.alphabet))

    def test_a_two_dimensional_reference_is_flattened(self):
        """An alphabet reshaped by a caller must not change the picture."""
        square = self.alphabet.reshape(4, 4)
        flat = plot_iq(self.received, reference=self.alphabet).lines[-1]
        shaped = plot_iq(self.received, reference=square).lines[-1]
        np.testing.assert_allclose(flat.get_xdata(), shaped.get_xdata())

    def test_a_reference_brings_a_legend(self):
        ax = plot_iq(self.received, reference=self.alphabet)
        self.assertIsNotNone(ax.get_legend())


class TestTheLabel(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.single = rng.standard_normal(50) + 1j * rng.standard_normal(50)
        self.pair = rng.standard_normal((2, 50)) + 1j * rng.standard_normal((2, 50))
        self.addCleanup(plt.close, "all")

    def test_a_single_stream_takes_the_label_as_given(self):
        ax = plot_iq(self.single, label="before")
        self.assertEqual(ax.lines[0].get_label(), "before")

    def test_two_signals_can_share_one_axis(self):
        """The pattern the fibre tutorial uses, before and after a fix."""
        ax = plot_iq(self.single, label="before")
        plot_iq(self.single + 0.1, label="after", ax=ax)
        self.assertEqual([line.get_label() for line in ax.lines],
                         ["before", "after"])

    def test_several_streams_are_numbered_under_the_label(self):
        ax = plot_iq(self.pair, label="rx")
        self.assertEqual([line.get_label() for line in ax.lines],
                         ["rx 0", "rx 1"])

    def test_without_a_label_streams_keep_their_default_names(self):
        ax = plot_iq(self.pair)
        self.assertEqual([line.get_label() for line in ax.lines],
                         ["stream 0", "stream 1"])


if __name__ == "__main__":
    unittest.main()
