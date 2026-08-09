r"""Achievable rates: MI, GMI and the normalized GMI (D48).

The estimators of Alvarado et al., JLT 33(20), 2015. What is checked
here is what the definitions *entail*, not what the code happens to
compute:

* the MI does not depend on how the constellation is labelled, and the
  GMI does -- that is the entire difference between a symbol-wise and a
  bit-wise decoder, and a GMI that ignored the labels would be a
  mislabelled MI;
* GMI <= MI always (the paper's eq. (24) and the discussion after it);
* both saturate at :math:`\log_2 M` in the noiseless limit and vanish
  in the noise-dominated one;
* max-log L-values support a *lower* rate, and the minimization over
  :math:`s` of eq. (32) is what keeps that statement honest -- without
  it the estimate would be lower still, and for the wrong reason.
"""
import unittest

import numpy as np

from comnumpy.core.information import (compute_gmi, compute_llr, compute_mi,
                                       compute_ngmi)
from comnumpy.core.mappers import SymbolDemapper
from comnumpy.core.utils import get_alphabet

N = 20000


def channel(alphabet, sigma2, seed=0, size=N):
    """Uniform symbols through a complex AWGN channel of variance sigma2."""
    rng = np.random.default_rng(seed)
    symbols = rng.integers(0, len(alphabet), size=size)
    noise = np.sqrt(sigma2 / 2) * (rng.normal(size=size)
                                   + 1j * rng.normal(size=size))
    return symbols, alphabet[symbols] + noise


class TestBounds(unittest.TestCase):

    def test_both_rates_saturate_at_log2_M(self):
        for order in (4, 16, 64):
            with self.subTest(M=order):
                alphabet = get_alphabet("QAM", order)
                symbols, received = channel(alphabet, 1e-6)
                bits = np.log2(order)
                self.assertAlmostEqual(compute_mi(received, symbols, alphabet),
                                       bits, places=6)
                self.assertAlmostEqual(compute_gmi(received, symbols, alphabet),
                                       bits, places=6)

    def test_both_rates_vanish_when_the_noise_dominates(self):
        alphabet = get_alphabet("QAM", 16)
        symbols, received = channel(alphabet, 400.0)
        self.assertLess(compute_mi(received, symbols, alphabet), 0.05)
        self.assertLess(compute_gmi(received, symbols, alphabet), 0.05)

    def test_the_gmi_never_exceeds_the_mi(self):
        alphabet = get_alphabet("QAM", 16)
        for sigma2 in (0.02, 0.1, 0.4, 1.0):
            with self.subTest(sigma2=sigma2):
                symbols, received = channel(alphabet, sigma2)
                self.assertLessEqual(
                    compute_gmi(received, symbols, alphabet),
                    compute_mi(received, symbols, alphabet) + 1e-9)

    def test_the_rates_decrease_with_the_noise(self):
        alphabet = get_alphabet("QAM", 16)
        rates = [compute_gmi(*channel(alphabet, s2)[::-1], alphabet)
                 for s2 in (0.02, 0.1, 0.4)]
        self.assertTrue(all(a > b for a, b in zip(rates, rates[1:],
                                                  strict=False)), rates)


class TestLabellingDependence(unittest.TestCase):
    """The property that separates the two rates.

    The mutual information is a property of the constellation and the
    channel; relabelling the points cannot change it. The GMI is a
    property of the *bit* channels, so a labelling that puts distant
    bits on nearby symbols destroys it.
    """

    def scrambled(self, alphabet):
        """A random relabelling: the same points, arbitrary bit labels."""
        return alphabet[np.random.default_rng(12).permutation(len(alphabet))]

    def test_the_mi_is_the_same_under_any_relabelling(self):
        alphabet = get_alphabet("QAM", 16)
        bad = self.scrambled(alphabet)
        symbols, _ = channel(alphabet, 0.1)
        rng = np.random.default_rng(7)
        noise = np.sqrt(0.05) * (rng.normal(size=len(symbols))
                                 + 1j * rng.normal(size=len(symbols)))
        gray_mi = compute_mi(alphabet[symbols] + noise, symbols, alphabet)
        bad_mi = compute_mi(bad[symbols] + noise, symbols, bad)
        # two estimates of the same quantity from two sequences:
        # equal in expectation, and this many samples is worth 0.005
        self.assertAlmostEqual(gray_mi, bad_mi, delta=0.02)

    def test_the_gmi_collapses_under_a_bad_labelling(self):
        alphabet = get_alphabet("QAM", 16)
        bad = self.scrambled(alphabet)
        symbols, _ = channel(alphabet, 0.1)
        rng = np.random.default_rng(7)
        noise = np.sqrt(0.05) * (rng.normal(size=len(symbols))
                                 + 1j * rng.normal(size=len(symbols)))
        gray = compute_gmi(alphabet[symbols] + noise, symbols, alphabet)
        scrambled = compute_gmi(bad[symbols] + noise, symbols, bad)
        self.assertGreater(gray - scrambled, 0.5)
        # ... while the MI is untouched: see the test above

    def test_gray_leaves_almost_no_gap_to_the_mi(self):
        """The reason Gray labelling is used at all."""
        alphabet = get_alphabet("QAM", 16)
        symbols, received = channel(alphabet, 0.1)
        gap = (compute_mi(received, symbols, alphabet)
               - compute_gmi(received, symbols, alphabet))
        self.assertLess(gap, 0.05)
        self.assertGreater(gap, 0.0)


class TestMaxLog(unittest.TestCase):
    r"""Eq. (32): the minimization over :math:`s` is mandatory."""

    def test_max_log_supports_a_lower_rate_than_exact_llrs(self):
        alphabet = get_alphabet("QAM", 16)
        symbols, received = channel(alphabet, 0.2)
        exact = compute_gmi(received, symbols, alphabet)
        approximate = compute_gmi(received, symbols, alphabet, max_log=True)
        self.assertLess(approximate, exact)
        self.assertGreater(approximate, exact - 0.1)

    def test_the_scaling_search_recovers_what_s_equal_to_one_loses(self):
        """Without the eq. (32) minimization the rate reads too low.

        The paper states it as a requirement rather than an option, so
        this pins the difference it makes rather than trusting the text.
        """
        alphabet = get_alphabet("QAM", 64)
        symbols, received = channel(alphabet, 0.05)
        llr = compute_llr(received, alphabet, snr=1 / 0.05, max_log=True)
        bits = (symbols[:, None] >> np.arange(5, -1, -1)) & 1
        signed = (1 - 2 * bits) * llr
        at_one = 6 - float(np.mean(np.sum(
            np.logaddexp(0.0, -signed), axis=-1)) / np.log(2))
        optimized = compute_gmi(received, symbols, alphabet, max_log=True)
        self.assertGreater(optimized, at_one)


class TestNormalizedGmi(unittest.TestCase):

    def test_it_is_the_gmi_over_the_bits_per_symbol(self):
        alphabet = get_alphabet("QAM", 64)
        symbols, received = channel(alphabet, 0.05)
        self.assertAlmostEqual(
            compute_ngmi(received, symbols, alphabet),
            compute_gmi(received, symbols, alphabet) / 6, places=12)

    def test_it_stays_between_zero_and_one(self):
        alphabet = get_alphabet("QAM", 16)
        for sigma2 in (1e-6, 0.1, 100.0):
            with self.subTest(sigma2=sigma2):
                symbols, received = channel(alphabet, sigma2)
                value = compute_ngmi(received, symbols, alphabet)
                self.assertGreaterEqual(value, -1e-9)
                self.assertLessEqual(value, 1 + 1e-9)


class TestLlr(unittest.TestCase):

    def test_the_sign_convention_matches_the_demapper(self):
        """One LLR convention in the library, not two."""
        alphabet = get_alphabet("QAM", 16)
        _, received = channel(alphabet, 0.05, size=200)
        sigma2 = 0.05
        theirs = SymbolDemapper(alphabet, soft=True, sigma2=sigma2)(received)
        mine = compute_llr(received, alphabet, snr=1 / sigma2, max_log=True)
        np.testing.assert_allclose(theirs.reshape(-1, 4), mine, rtol=1e-9)

    def test_the_exact_llrs_agree_with_max_log_at_high_snr(self):
        alphabet = get_alphabet("QAM", 16)
        _, received = channel(alphabet, 1e-4, size=500)
        exact = compute_llr(received, alphabet, snr=1e4)
        approximate = compute_llr(received, alphabet, snr=1e4, max_log=True)
        relative = (np.abs(exact - approximate)
                    / np.maximum(np.abs(exact), 1e-12))
        self.assertLess(float(np.max(relative)), 1e-3)

    def test_an_alphabet_that_is_not_a_power_of_two_is_refused(self):
        hexagonal = np.exp(2j * np.pi * np.arange(6) / 6)
        with self.assertRaises(ValueError) as ctx:
            compute_gmi(np.zeros(6, dtype=complex), np.arange(6), hexagonal)
        self.assertIn("power of two", str(ctx.exception))
        self.assertIn("compute_mi", str(ctx.exception))


class TestAuxiliaryChannel(unittest.TestCase):

    def test_the_estimated_snr_reproduces_the_true_one(self):
        alphabet = get_alphabet("QAM", 16)
        symbols, received = channel(alphabet, 0.1, size=200000)
        explicit = compute_mi(received, symbols, alphabet,
                              snr=np.mean(np.abs(alphabet) ** 2) / 0.1)
        estimated = compute_mi(received, symbols, alphabet)
        self.assertAlmostEqual(explicit, estimated, places=3)

    def test_a_noiseless_observation_says_what_to_pass(self):
        alphabet = get_alphabet("QAM", 4)
        symbols = np.arange(4)
        with self.assertRaises(ValueError) as ctx:
            compute_mi(alphabet[symbols], symbols, alphabet)
        self.assertIn("snr=", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
