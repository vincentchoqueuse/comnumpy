"""Pulse shaping over the axes the rest of the library actually uses.

``SRRCFilter``'s frequency-domain path used to size its FFT with
``len(x)``, which is the number of *rows* of a two-dimensional array, not
the number of samples. A polarization pair ``(..., 2, N)`` -- the shape
D47 introduced for the Manakov equation -- therefore transformed two
samples instead of N and died inside ``H()`` with a message about
negative dimensions that named neither the filter nor the shape.

Both methods now filter along the last axis and broadcast over the
leading ones. The tests below pin the two properties that matter: the
two paths agree, and shaping a stack is shaping each row.
"""
import unittest

import numpy as np

from comnumpy.core.filters import SRRCFilter

OVERSAMPLING = 4
N_SYM = 512


def symbols(rng, shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def upsample(x):
    out = np.zeros(x.shape[:-1] + (x.shape[-1] * OVERSAMPLING,), dtype=complex)
    out[..., ::OVERSAMPLING] = x
    return out


class TestTheFftPathBroadcasts(unittest.TestCase):

    def setUp(self):
        self.filter = SRRCFilter(0.1, OVERSAMPLING, N_h=8, method="fft")
        self.rng = np.random.default_rng(0)

    def test_a_polarization_pair_is_each_polarization(self):
        stack = upsample(symbols(self.rng, (2, N_SYM)))
        filtered = self.filter(stack)
        self.assertEqual(filtered.shape, stack.shape)
        for row in range(2):
            with self.subTest(row=row):
                np.testing.assert_allclose(filtered[row],
                                           self.filter(stack[row]), atol=1e-12)

    def test_any_leading_shape_broadcasts(self):
        for shape in ((3, N_SYM), (2, 2, N_SYM), (1, 5, N_SYM)):
            with self.subTest(shape=shape):
                stack = upsample(symbols(self.rng, shape))
                filtered = self.filter(stack)
                self.assertEqual(filtered.shape, stack.shape)
                flat = stack.reshape(-1, stack.shape[-1])
                for index in range(flat.shape[0]):
                    np.testing.assert_allclose(
                        filtered.reshape(-1, stack.shape[-1])[index],
                        self.filter(flat[index]), atol=1e-12)

    def test_the_one_dimensional_answer_did_not_move(self):
        """The fix must be invisible to every existing 1D caller."""
        signal = upsample(symbols(self.rng, (N_SYM,)))
        spectrum = np.fft.fft(signal, signal.size)
        expected = np.fft.ifft(spectrum * self.filter.H(signal.size),
                               signal.size)
        np.testing.assert_allclose(self.filter(signal), expected, atol=1e-12)


class TestMatchedFilteringIsIsiFree(unittest.TestCase):
    """Two root-raised cosines make a raised cosine: no ISI at the peaks."""

    def test_a_pair_of_polarizations_round_trips(self):
        rng = np.random.default_rng(1)
        sent = symbols(rng, (2, N_SYM))
        shaper = SRRCFilter(0.1, OVERSAMPLING, N_h=40, method="fft")
        matched = SRRCFilter(0.1, OVERSAMPLING, N_h=40, method="fft",
                             scale=1 / OVERSAMPLING)
        received = matched(shaper(upsample(sent) * OVERSAMPLING))
        sampled = received[..., ::OVERSAMPLING]
        gain = np.vdot(sent, sampled) / np.vdot(sent, sent)
        residual = np.mean(np.abs(sampled - gain * sent) ** 2)
        self.assertLess(residual / np.mean(np.abs(gain * sent) ** 2), 1e-6)


if __name__ == "__main__":
    unittest.main()
