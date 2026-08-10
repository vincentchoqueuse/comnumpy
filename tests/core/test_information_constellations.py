r"""The Monte-Carlo rates against a quadrature computed the other way.

:mod:`comnumpy.core.information` estimates the MI and the GMI from a
*record*: symbols that were sent, samples that came back, the estimators
of Alvarado et al. (2015). :mod:`comnumpy.core.capacity` computes the
same two quantities for the AWGN channel from the constellation alone,
by Gauss-Hermite quadrature -- no random draws at all. Their definitions
agree by construction:

* eq. (16)-(17), the MI, is the constellation-constrained capacity
  :func:`~comnumpy.core.capacity.constellation_capacity`;
* eq. (21)-(26), the GMI, is :math:`\sum_i I(B_i; Y)`, which is the
  literal definition of
  :func:`~comnumpy.core.capacity.bicm_capacity`.

So the two modules must produce the same numbers, and they were written
independently, from different papers, with different numerics. That is
what this file exercises -- over eight constellations of three families
and three SNRs each, rather than the single 16-QAM the rest of the suite
uses.

The sweep also pins three facts about the *modulations* that a
single-constellation test cannot show: which of them make the bit-wise
interface free, when it stops being free, and that the ordering by
constellation size is the expected one.
"""
import unittest

import numpy as np

from comnumpy.core.capacity import bicm_capacity, constellation_capacity
from comnumpy.core.information import compute_gmi, compute_mi
from comnumpy.core.utils import get_alphabet

N = 40000

# Three families, so that the comparison is not a property of one
# geometry: constant modulus (PSK), one-dimensional (PAM), square (QAM).
CONSTELLATIONS = [("PSK", 2), ("PSK", 4), ("PSK", 8),
                  ("PAM", 4), ("PAM", 8),
                  ("QAM", 4), ("QAM", 16), ("QAM", 64)]

SNR_DB = [0.0, 8.0, 16.0]

# 40000 symbols are worth about 0.015 bit on the hardest row (64-QAM,
# where the estimator averages over 64 hypotheses); raising the record to
# 200000 brings the same rows under 0.001 bit, which is how we know the
# residual is the Monte-Carlo draw and not the quadrature. Keep the
# tolerance at twice the observed spread.
TOLERANCE = 0.03


def channel(alphabet, snr, seed=0, size=N):
    """Uniform symbols through a complex AWGN channel of the given SNR.

    The alphabet has unit average energy, so the noise variance is
    :math:`\\sigma^2 = 1/\\rho` and both modules are given the same
    :math:`\\rho`.
    """
    rng = np.random.default_rng(seed)
    symbols = rng.integers(0, len(alphabet), size=size)
    sigma2 = 1.0 / snr
    noise = np.sqrt(sigma2 / 2) * (rng.normal(size=size)
                                   + 1j * rng.normal(size=size))
    return symbols, alphabet[symbols] + noise


class TestAgainstTheQuadrature(unittest.TestCase):
    """Two implementations, one number -- eight times over, twice each."""

    def test_the_mi_matches_the_constellation_capacity(self):
        for family, order in CONSTELLATIONS:
            alphabet = get_alphabet(family, order)
            for snr_dB in SNR_DB:
                with self.subTest(constellation=f"{family}-{order}",
                                  snr_dB=snr_dB):
                    snr = 10 ** (snr_dB / 10)
                    symbols, received = channel(alphabet, snr)
                    estimated = compute_mi(received, symbols, alphabet,
                                           snr=snr)
                    exact = float(constellation_capacity(alphabet, snr))
                    self.assertAlmostEqual(estimated, exact,
                                           delta=TOLERANCE)

    def test_the_gmi_matches_the_bicm_capacity(self):
        for family, order in CONSTELLATIONS:
            alphabet = get_alphabet(family, order)
            for snr_dB in SNR_DB:
                with self.subTest(constellation=f"{family}-{order}",
                                  snr_dB=snr_dB):
                    snr = 10 ** (snr_dB / 10)
                    symbols, received = channel(alphabet, snr)
                    estimated = compute_gmi(received, symbols, alphabet,
                                            snr=snr)
                    exact = float(bicm_capacity(alphabet, snr))
                    self.assertAlmostEqual(estimated, exact,
                                           delta=TOLERANCE)

    def test_the_deviation_shrinks_with_the_record_length(self):
        """The residual is the estimator's variance, not a bias.

        64-QAM at 8 dB is the worst row of the sweep above -- the
        estimator averages over 64 hypotheses there. If the two modules
        disagreed on the *quantity*, a longer record would not help. It
        does, so what separates them is the draw.

        One record is not evidence, since a single draw can land on the
        right answer by luck: the deviation is measured in RMS over
        several independent records at each length.
        """
        alphabet = get_alphabet("QAM", 64)
        snr = 10 ** 0.8
        exact = float(constellation_capacity(alphabet, snr))
        spread = []
        for size in (2000, 100000):
            deviations = [
                compute_mi(*channel(alphabet, snr, seed=seed, size=size)[::-1],
                           alphabet, snr=snr) - exact
                for seed in range(4)]
            spread.append(float(np.sqrt(np.mean(np.square(deviations)))))
        self.assertLess(spread[1], spread[0] / 2)


class TestWhatTheFamiliesDiffer(unittest.TestCase):
    """What eight constellations show that one cannot."""

    def test_binary_and_quaternary_labellings_make_the_two_rates_equal(self):
        r"""Two bits carried by two orthogonal dimensions cost nothing.

        For BPSK there is one bit per symbol, so :math:`I(B_1; Y)` *is*
        the MI. For QPSK -- and 4-QAM, which is the same constellation --
        the two bits ride the two quadratures, which the complex AWGN
        channel keeps independent: the bit-wise interface throws nothing
        away and the GMI equals the MI exactly.
        """
        for family, order in [("PSK", 2), ("PSK", 4), ("QAM", 4)]:
            alphabet = get_alphabet(family, order)
            for snr_dB in SNR_DB:
                with self.subTest(constellation=f"{family}-{order}",
                                  snr_dB=snr_dB):
                    snr = 10 ** (snr_dB / 10)
                    symbols, received = channel(alphabet, snr)
                    self.assertAlmostEqual(
                        compute_mi(received, symbols, alphabet, snr=snr),
                        compute_gmi(received, symbols, alphabet, snr=snr),
                        places=9)

    def test_the_bicm_gap_opens_beyond_four_points(self):
        """And it is largest where the decoder is least sure.

        Eight or more points cannot be labelled so that the bits stay
        independent, so the bit-wise interface starts costing something.
        The cost is a low-SNR phenomenon: at high SNR the demapper is
        rarely in doubt, and the gap closes again.
        """
        for family, order in [("PSK", 8), ("PAM", 8), ("QAM", 16),
                              ("QAM", 64)]:
            alphabet = get_alphabet(family, order)
            gaps = []
            for snr_dB in SNR_DB:
                snr = 10 ** (snr_dB / 10)
                symbols, received = channel(alphabet, snr)
                gaps.append(compute_mi(received, symbols, alphabet, snr=snr)
                            - compute_gmi(received, symbols, alphabet,
                                          snr=snr))
            with self.subTest(constellation=f"{family}-{order}"):
                self.assertGreater(gaps[0], 0.02)          # visible at 0 dB
                self.assertGreater(gaps[0], gaps[-1])      # gone at 16 dB
                self.assertLess(gaps[-1], 0.05)

    def test_no_constellation_ever_exceeds_the_shannon_capacity(self):
        """A discrete input cannot beat a Gaussian one."""
        from comnumpy.core.capacity import awgn_capacity
        for family, order in CONSTELLATIONS:
            alphabet = get_alphabet(family, order)
            for snr_dB in SNR_DB:
                with self.subTest(constellation=f"{family}-{order}",
                                  snr_dB=snr_dB):
                    snr = 10 ** (snr_dB / 10)
                    symbols, received = channel(alphabet, snr)
                    ceiling = float(awgn_capacity(snr))
                    self.assertLessEqual(
                        compute_mi(received, symbols, alphabet, snr=snr),
                        ceiling + TOLERANCE)

    def test_a_larger_constellation_carries_more_at_the_same_snr(self):
        """Within a family, and only up to the saturation of the small one.

        At 16 dB the MI of QPSK is pinned at its 2 bit ceiling while
        16-QAM and 64-QAM are still climbing -- which is the whole reason
        to change modulation format with the link budget.
        """
        snr = 10 ** 1.6
        rates = []
        for order in (4, 16, 64):
            alphabet = get_alphabet("QAM", order)
            symbols, received = channel(alphabet, snr)
            rates.append(compute_mi(received, symbols, alphabet, snr=snr))
        self.assertTrue(all(a < b for a, b in zip(rates, rates[1:],
                                                  strict=False)), rates)
        self.assertAlmostEqual(rates[0], 2.0, delta=1e-3)   # QPSK saturated

    def test_one_dimensional_formats_pay_for_the_unused_quadrature(self):
        """8-PAM against 8-PSK: the same rate ceiling, not the same rate.

        Both carry three bits, but PAM spends its energy on one axis
        while the noise fills two, so at a given SNR it achieves less.
        """
        snr = 10 ** 0.8
        rates = {}
        for family in ("PAM", "PSK"):
            alphabet = get_alphabet(family, 8)
            symbols, received = channel(alphabet, snr)
            rates[family] = compute_mi(received, symbols, alphabet, snr=snr)
        self.assertLess(rates["PAM"], rates["PSK"])


if __name__ == "__main__":
    unittest.main()
