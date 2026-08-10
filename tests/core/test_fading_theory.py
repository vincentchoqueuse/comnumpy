r"""Closed-form error rates over Rayleigh fading, and what pins them.

The AWGN expressions of :mod:`comnumpy.core.metrics` had no fading
counterpart, so a MIMO or diversity simulation had nothing analytical to
be read against -- the library could only compare one Monte-Carlo curve
with another. These are that counterpart, and they are checked three
ways rather than trusted:

* against a **closed form**: Proakis gives the L-branch BPSK error rate
  as a finite sum, and the quadrature must reproduce it to the last
  digits;
* against the **defining property**: the error rate decays as
  ``SNR^-L``, so the slope on a log-log plot *is* the diversity order;
* against a **simulation of the library's own blocks**: a chain built
  from ``FlatMIMOChannel``, ``AWGN`` and a detector must land on the
  curve, which is the only check that the *convention* -- what "SNR per
  branch" means, where the power split goes -- is the same on both
  sides.
"""
import unittest
from math import comb

import numpy as np

from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import (compute_metric_rayleigh_theo,
                                   compute_ser_rayleigh_psk,
                                   compute_ser_rayleigh_qam)
from comnumpy.core.processors import Amplifier
from comnumpy.core.utils import get_alphabet
from comnumpy.mimo.channels import AWGN, FlatMIMOChannel
from comnumpy.mimo.coding import SpaceTimeDecoder, SpaceTimeEncoder, get_code
from comnumpy.mimo.detectors import LinearDetector
from comnumpy.mimo.utils import rayleigh_channel

SNR_DB = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
SNR = 10 ** (SNR_DB / 10)


def proakis_bpsk(snr_per_bit, diversity):
    r"""Proakis 5th ed., Section 13.4: the L-branch BPSK error rate.

    .. math::

        P_b = \left[\frac{1-\mu}{2}\right]^{L}
              \sum_{k=0}^{L-1} \binom{L-1+k}{k}
              \left[\frac{1+\mu}{2}\right]^{k},
        \qquad \mu = \sqrt{\frac{\bar\gamma}{1+\bar\gamma}}
    """
    mu = np.sqrt(snr_per_bit / (1 + snr_per_bit))
    return ((1 - mu) / 2) ** diversity * sum(
        comb(diversity - 1 + k, k) * ((1 + mu) / 2) ** k
        for k in range(diversity))


class TestAgainstTheClosedForm(unittest.TestCase):

    def test_bpsk_reproduces_proakis(self):
        for diversity in (1, 2, 3, 4, 8):
            with self.subTest(diversity=diversity):
                quadrature = compute_ser_rayleigh_psk(2, SNR,
                                                      diversity=diversity)
                closed = proakis_bpsk(SNR, diversity)
                np.testing.assert_allclose(quadrature, closed, rtol=1e-10)

    def test_the_single_branch_case_is_the_textbook_one_liner(self):
        """L = 1 collapses to (1 - sqrt(g/(1+g)))/2."""
        expected = 0.5 * (1 - np.sqrt(SNR / (1 + SNR)))
        np.testing.assert_allclose(compute_ser_rayleigh_psk(2, SNR), expected,
                                   rtol=1e-10)

    def test_qpsk_sits_between_one_and_two_bpsk_quadratures(self):
        r"""The AWGN identity does **not** survive the fading average.

        Conditioned on the channel,
        :math:`P_s = 1 - (1 - P_b)^2`; averaging both sides over
        :math:`|h|^2` breaks it, because the mean of a square is not the
        square of the mean. What survives is the union bound, which
        holds conditionally and therefore on average.
        """
        bpsk = compute_ser_rayleigh_psk(2, SNR)
        qpsk = compute_ser_rayleigh_psk(4, SNR)
        self.assertTrue(np.all(qpsk <= 2 * bpsk))
        self.assertTrue(np.all(qpsk >= bpsk))
        # and it is *not* the AWGN identity: that would give this ratio
        self.assertFalse(np.allclose(qpsk, 1 - (1 - bpsk) ** 2, rtol=1e-3))

    def test_4qam_and_qpsk_are_the_same_constellation(self):
        for diversity in (1, 2, 4):
            with self.subTest(diversity=diversity):
                np.testing.assert_allclose(
                    compute_ser_rayleigh_qam(4, SNR, diversity=diversity),
                    compute_ser_rayleigh_psk(4, SNR, diversity=diversity),
                    rtol=1e-9)


class TestDiversityOrder(unittest.TestCase):
    r"""The exponent is the whole point: :math:`P_e \propto \gamma^{-L}`."""

    def slope(self, values, snr_dB):
        return float(np.polyfit(snr_dB / 10, np.log10(values), 1)[0])

    def test_the_high_snr_slope_is_the_number_of_branches(self):
        high = np.array([30.0, 35.0, 40.0])
        for diversity in (1, 2, 3, 4):
            for family, order in (("PSK", 2), ("PSK", 8), ("QAM", 16)):
                with self.subTest(diversity=diversity,
                                  modulation=f"{family}-{order}"):
                    values = compute_metric_rayleigh_theo(
                        family, order, 10 ** (high / 10), diversity=diversity)
                    self.assertAlmostEqual(self.slope(values, high),
                                           -diversity, delta=0.02)

    def test_more_branches_never_hurt(self):
        previous = compute_ser_rayleigh_qam(16, SNR, diversity=1)
        for diversity in (2, 3, 4):
            current = compute_ser_rayleigh_qam(16, SNR, diversity=diversity)
            with self.subTest(diversity=diversity):
                self.assertTrue(np.all(current <= previous))
            previous = current

    def test_fading_is_always_worse_than_awgn(self):
        """Jensen, in one line: averaging the error rate costs."""
        from comnumpy.core.metrics import compute_ser_awgn_psk
        self.assertTrue(np.all(compute_ser_rayleigh_psk(4, SNR)
                               > compute_ser_awgn_psk(4, SNR)))


class TestAgainstTheLibrarysOwnChains(unittest.TestCase):
    """The convention check: does a simulated link land on the curve?

    A closed form is only useful if the library and the formula agree on
    what the SNR *is*. Three chains, three readings of the same
    expression, and the numbers have to meet.
    """

    # The Monte-Carlo error over fading is set by the number of
    # *channel* draws, not by the symbol count: the error rate is
    # dominated by the rare deep fades. 4000 draws is what a unit test
    # can afford, and it is only trustworthy while the error rate stays
    # above ~1e-2 -- at 12 dB and 4000 draws the estimate already reads
    # 16 % low. The high-SNR confrontation lives in validation/ (D7),
    # where the draw count can be what the claim needs.
    ORDER = 4
    N_CHANNELS = 4000
    N_SYMBOLS = 40
    SNR_DB = (4.0, 8.0)
    TOLERANCE = 0.15

    def alphabet(self):
        return get_alphabet("PSK", self.ORDER)

    def simulate(self, chain, n_rx, n_tx, snr_dB, stimulus, seed=0):
        rng = np.random.default_rng(seed)
        chain.seed(seed)
        chain.set_params(**{"noise.sigma2": 10 ** (-snr_dB / 10)})
        errors = total = 0
        for _ in range(self.N_CHANNELS):
            realization = rayleigh_channel(n_rx, n_tx, rng=rng)
            chain.set_params(**{"channel.H": realization,
                                "detector.H": realization})
            detected = chain(stimulus)
            errors += int(np.sum(chain.tap("tx") != detected))
            total += detected.size
        return errors / total

    def linear_chain(self, H):
        alphabet = self.alphabet()
        return Sequential([
            SymbolGenerator(self.ORDER, name="tx"),
            SymbolMapper(alphabet),
            FlatMIMOChannel(H, name="channel"),
            AWGN(sigma2=0.1, name="noise"),
            LinearDetector(alphabet, H=H, name="detector"),
        ], taps=["tx"])

    def test_one_antenna_each_side_matches_diversity_one(self):
        chain = self.linear_chain(rayleigh_channel(1, 1, seed=0))
        for snr_dB in self.SNR_DB:
            measured = self.simulate(chain, 1, 1, snr_dB, (1, self.N_SYMBOLS))
            theory = compute_ser_rayleigh_psk(
                self.ORDER, 10 ** (snr_dB / 10) / np.log2(self.ORDER))
            with self.subTest(snr_dB=snr_dB):
                self.assertAlmostEqual(measured, float(theory),
                                       delta=self.TOLERANCE * float(theory))

    def test_receive_diversity_matches_diversity_n_rx(self):
        """Zero forcing on an (N_r, 1) channel is maximum ratio combining."""
        chain = self.linear_chain(rayleigh_channel(2, 1, seed=0))
        for snr_dB in self.SNR_DB:
            measured = self.simulate(chain, 2, 1, snr_dB, (1, self.N_SYMBOLS))
            theory = compute_ser_rayleigh_psk(
                self.ORDER, 10 ** (snr_dB / 10) / np.log2(self.ORDER),
                diversity=2)
            with self.subTest(snr_dB=snr_dB):
                self.assertAlmostEqual(measured, float(theory),
                                       delta=self.TOLERANCE * float(theory))

    def test_alamouti_matches_two_branches_at_half_the_branch_snr(self):
        r"""The reading that costs 3 dB, checked rather than asserted.

        A 2x1 orthogonal design is a two-branch combiner whose branches
        each carry half the transmit power, so it is
        ``diversity=2`` evaluated at :math:`\bar\gamma / N_t`. If that
        division were forgotten the curve would sit 3 dB to the left,
        which 20 % tolerance cannot hide.
        """
        alphabet = self.alphabet()
        code = get_code("alamouti")
        power = 1 / np.sqrt(code.n_tx)
        H = rayleigh_channel(1, 2, seed=0)
        chain = Sequential([
            SymbolGenerator(self.ORDER, name="tx"),
            SymbolMapper(alphabet),
            Amplifier(power),
            SpaceTimeEncoder(code),
            FlatMIMOChannel(H, name="channel"),
            AWGN(sigma2=0.1, name="noise"),
            SpaceTimeDecoder(code, H=H, name="detector"),
            Amplifier(1 / power),
            SymbolDemapper(alphabet),
        ], taps=["tx"])
        for snr_dB in self.SNR_DB:
            measured = self.simulate(chain, 1, 2, snr_dB, self.N_SYMBOLS)
            theory = compute_ser_rayleigh_psk(
                self.ORDER,
                10 ** (snr_dB / 10) / np.log2(self.ORDER) / code.n_tx,
                diversity=2)
            with self.subTest(snr_dB=snr_dB):
                self.assertAlmostEqual(measured, float(theory),
                                       delta=self.TOLERANCE * float(theory))


class TestGuards(unittest.TestCase):

    def test_a_non_square_qam_is_refused_towards_the_psk_function(self):
        with self.assertRaises(ValueError) as ctx:
            compute_ser_rayleigh_qam(8, 10.0)
        self.assertIn("compute_ser_rayleigh_psk", str(ctx.exception))

    def test_zero_branches_is_refused(self):
        with self.assertRaises(ValueError):
            compute_ser_rayleigh_psk(4, 10.0, diversity=0)

    def test_an_unknown_modulation_names_the_two_families(self):
        with self.assertRaises(ValueError) as ctx:
            compute_metric_rayleigh_theo("FSK", 4, 10.0)
        self.assertIn("'PSK' or 'QAM'", str(ctx.exception))

    def test_a_scalar_in_gives_a_scalar_out(self):
        value = compute_ser_rayleigh_psk(4, 10.0)
        self.assertIsInstance(value, float)


if __name__ == "__main__":
    unittest.main()
