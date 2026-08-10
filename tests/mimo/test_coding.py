r"""Space-time codes: the properties that define them, not the matrices.

A space-time block code is not a table of symbols; it is a claim about
three numbers -- its rate, its diversity order and its decoding cost --
and each of those is a checkable statement:

* **rate** :math:`K/T`, the symbols carried per channel use;
* **diversity**, the minimum rank of a difference codeword (Tarokh,
  Seshadri, Calderbank 1998), which shows up as the *slope* of the error
  curve and is measured that way here;
* **decoding**, exact by matched filter for an orthogonal design, and
  only for one -- which is what the orthogonality identity
  :math:`\mathbf{M}^{\mathsf{T}}\mathbf{M} = c\|\mathbf{H}\|_F^2
  \mathbf{I}` buys, and why every code declaring itself orthogonal is
  made to prove it at construction.

The trade the whole subject is about is checked directly: over the codes
in the registry, rate and diversity cannot both be maximal.
"""
import unittest

import numpy as np

from comnumpy.core.utils import get_alphabet, hard_projector
from comnumpy.exceptions import ShapeError
from comnumpy.mimo.coding import (SpaceTimeCode, SpaceTimeDecoder,
                                  SpaceTimeEncoder, available_codes,
                                  coding_gain, get_code, register_code)

ORTHOGONAL = ["alamouti", "ostbc-3-1/2", "ostbc-4-1/2",
              "ostbc-3-3/4", "ostbc-4-3/4"]
QPSK = get_alphabet("QAM", 4)


def channel(n_rx, n_tx, rng):
    shape = (n_rx, n_tx)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)


class TestOrthogonality(unittest.TestCase):
    r"""The identity that makes linear decoding optimal."""

    def test_the_declared_codes_satisfy_it(self):
        rng = np.random.default_rng(0)
        for name in ORTHOGONAL:
            code = get_code(name)
            for n_rx in (1, 2, 4):
                H = channel(n_rx, code.n_tx, rng)
                matrix = code.equivalent_channel(H)
                gram = matrix.T @ matrix
                expected = (code.orthogonality_gain
                            * float(np.sum(np.abs(H) ** 2))
                            * np.eye(2 * code.n_symbols))
                with self.subTest(code=name, n_rx=n_rx):
                    np.testing.assert_allclose(gram, expected, atol=1e-10)

    def test_a_code_that_lies_about_it_is_refused_at_construction(self):
        """D20: the guard that caught a wrong sign while this was written."""
        direct = np.zeros((2, 2, 2), dtype=complex)
        direct[0, 0, 0] = direct[1, 1, 0] = 1.0      # not an orthogonal design
        with self.assertRaises(ValueError) as ctx:
            SpaceTimeCode("liar", direct, np.zeros_like(direct),
                          orthogonal=True)
        message = str(ctx.exception)
        self.assertIn("declared orthogonal", message)
        self.assertIn("conjugations", message)

    def test_the_gain_is_one_for_alamouti_and_two_for_the_half_rate_designs(self):
        """It is measured, not assumed: the decoder divides by it."""
        self.assertAlmostEqual(get_code("alamouti").orthogonality_gain, 1.0)
        self.assertAlmostEqual(get_code("ostbc-3-3/4").orthogonality_gain, 1.0)
        for name in ("ostbc-3-1/2", "ostbc-4-1/2"):
            with self.subTest(code=name):
                self.assertAlmostEqual(get_code(name).orthogonality_gain, 2.0)


class TestEncodeDecode(unittest.TestCase):

    def test_a_noiseless_link_returns_the_symbols_exactly(self):
        rng = np.random.default_rng(1)
        alphabet = get_alphabet("QAM", 16)
        for name in ORTHOGONAL:
            code = get_code(name)
            for n_rx in (1, 2, 3):
                symbols = alphabet[rng.integers(0, 16,
                                                size=5 * code.n_symbols)]
                H = channel(n_rx, code.n_tx, rng)
                received = H @ SpaceTimeEncoder(code)(symbols)
                estimate = SpaceTimeDecoder(code, H=H)(received)
                with self.subTest(code=name, n_rx=n_rx):
                    np.testing.assert_allclose(estimate, symbols, atol=1e-12)

    def test_the_encoder_changes_the_number_of_channel_uses(self):
        """A rate-1/2 code spends twice the slots; that is the trade."""
        for name in ORTHOGONAL:
            code = get_code(name)
            symbols = np.ones(3 * code.n_symbols, dtype=complex)
            with self.subTest(code=name):
                self.assertEqual(SpaceTimeEncoder(code)(symbols).shape,
                                 (code.n_tx, 3 * code.n_slots))

    def test_the_matched_filter_is_maximum_likelihood(self):
        r"""Not an approximation: it matches an exhaustive ML search.

        For an orthogonal design the two coincide *sample by sample*,
        including where the noise makes them both wrong.
        """
        rng = np.random.default_rng(2)
        code = get_code("alamouti")
        H = channel(2, 2, rng)
        for _ in range(60):
            symbols = QPSK[rng.integers(0, 4, size=2)]
            noise = 0.7 * (rng.normal(size=(2, 2))
                           + 1j * rng.normal(size=(2, 2)))
            received = H @ code.encode(symbols) + noise
            linear, _ = hard_projector(SpaceTimeDecoder(code, H=H)(received),
                                       QPSK)
            self.assertEqual(tuple(linear), tuple(_exhaustive_ml(code, H,
                                                                 received)))

    def test_a_non_orthogonal_code_is_refused_with_the_way_out(self):
        with self.assertRaises(ValueError) as ctx:
            SpaceTimeDecoder(get_code("golden"))
        message = str(ctx.exception)
        self.assertIn("equivalent_channel", message)
        self.assertIn("comnumpy.mimo.detectors", message)

    def test_a_missing_channel_says_how_to_supply_it(self):
        decoder = SpaceTimeDecoder(get_code("alamouti"))
        with self.assertRaises(ValueError) as ctx:
            decoder(np.zeros((1, 2), dtype=complex))
        self.assertIn("decoder.H", str(ctx.exception))

    def test_a_truncated_block_is_refused(self):
        code = get_code("ostbc-3-1/2")
        with self.assertRaises(ShapeError):
            SpaceTimeEncoder(code)(np.ones(code.n_symbols + 1, dtype=complex))


def _exhaustive_ml(code, H, received):
    """The definition: minimize the distance over every symbol vector."""
    best, argbest = np.inf, None
    for first in range(len(QPSK)):
        for second in range(len(QPSK)):
            candidate = QPSK[[first, second]]
            error = received - H @ code.encode(candidate)
            value = float(np.sum(np.abs(error) ** 2))
            if value < best:
                best, argbest = value, (first, second)
    return argbest


class TestDiversityAndRate(unittest.TestCase):

    def test_the_orthogonal_designs_are_full_diversity(self):
        for name in ORTHOGONAL:
            code = get_code(name)
            with self.subTest(code=name):
                self.assertEqual(code.minimum_rank(QPSK), code.n_tx)

    def test_spatial_multiplexing_has_no_diversity_at_all(self):
        """Rate n_tx, rank 1: the opposite corner of the same trade."""
        code = get_code("spatial-multiplexing", n_tx=4)
        self.assertEqual(code.rate, 4.0)
        self.assertEqual(code.minimum_rank(QPSK), 1)
        self.assertEqual(coding_gain(code, QPSK), 0.0)

    def test_the_golden_code_is_full_rate_and_full_diversity(self):
        """What it was built for, and what an OSTBC cannot do at 2x2."""
        code = get_code("golden")
        self.assertEqual(code.rate, 2.0)
        self.assertEqual(code.minimum_rank(QPSK), 2)
        self.assertGreater(coding_gain(code, QPSK), 0.0)
        # ...and it is not orthogonal: no 2x2 code is, above rate 1
        self.assertFalse(code.is_orthogonal)

    def test_rate_and_diversity_cannot_both_be_maximal(self):
        """The trade, over the whole registry."""
        for name in available_codes():
            code = get_code(name)
            with self.subTest(code=name):
                full_diversity = code.minimum_rank(QPSK) == code.n_tx
                self.assertTrue(code.rate <= 1.0 or not code.is_orthogonal)
                if code.rate > 1.0:
                    # above rate 1 nothing is orthogonal, so nothing
                    # decodes linearly -- diversity has to be bought back
                    self.assertFalse(code.is_orthogonal)
                elif full_diversity and code.is_orthogonal:
                    self.assertLessEqual(code.rate, 1.0)

    def test_the_declared_rates_are_the_published_ones(self):
        expected = {"alamouti": 1.0, "ostbc-3-1/2": 0.5, "ostbc-4-1/2": 0.5,
                    "ostbc-3-3/4": 0.75, "ostbc-4-3/4": 0.75,
                    "golden": 2.0, "spatial-multiplexing": 2.0}
        for name, rate in expected.items():
            with self.subTest(code=name):
                self.assertEqual(get_code(name).rate, rate)


class TestDiversityIsVisible(unittest.TestCase):
    r"""Diversity is a *slope*, so it is measured as one.

    The symbol error rate of a scheme of diversity order :math:`d`
    decays as :math:`\mathrm{SNR}^{-d}`, i.e. by :math:`10d` dB per
    decade. Alamouti on two transmit antennas against one receive
    antenna must show slope 2, a single antenna slope 1 -- and the
    comparison only means something because both spend the **same total
    transmit power**: the codeword is scaled by
    :math:`1/\sqrt{N_t}`, since two antennas each sending :math:`|s|^2`
    would be a 3 dB advantage before any coding.

    The channel is drawn once per block of 40 codewords (quasi-static
    fading), which is both the standard model for this code and what
    keeps the measurement affordable.
    """

    N_CHANNELS = 4000
    PER_CHANNEL = 40
    SNR_DB = (13.0, 19.0, 25.0)

    def slope(self, errors):
        """Least-squares slope of log10(SER) against SNR/10 in dB."""
        return float(np.polyfit(np.array(self.SNR_DB) / 10,
                                np.log10(errors), 1)[0])

    def alamouti_ser(self, snr_dB, seed=3):
        rng = np.random.default_rng(seed)
        code = get_code("alamouti")
        encoder = SpaceTimeEncoder(code)
        power = 1 / np.sqrt(code.n_tx)          # same total power as SISO
        sigma2 = 10 ** (-snr_dB / 10)
        errors = total = 0
        for _ in range(self.N_CHANNELS):
            H = channel(1, 2, rng)
            sent = rng.integers(0, 4, size=2 * self.PER_CHANNEL)
            word = encoder(QPSK[sent] * power)
            noise = np.sqrt(sigma2 / 2) * (
                rng.normal(size=(1, word.shape[1]))
                + 1j * rng.normal(size=(1, word.shape[1])))
            estimate = SpaceTimeDecoder(code, H=H)(H @ word + noise) / power
            detected, _ = hard_projector(estimate, QPSK)
            errors += int(np.sum(detected != sent))
            total += sent.size
        return errors / total

    def siso_ser(self, snr_dB, seed=4):
        """One antenna, same total transmit power: diversity 1."""
        rng = np.random.default_rng(seed)
        size = self.N_CHANNELS * self.PER_CHANNEL
        sent = rng.integers(0, 4, size=size)
        h = (rng.normal(size=size) + 1j * rng.normal(size=size)) / np.sqrt(2)
        sigma2 = 10 ** (-snr_dB / 10)
        noise = np.sqrt(sigma2 / 2) * (rng.normal(size=size)
                                       + 1j * rng.normal(size=size))
        detected, _ = hard_projector((h * QPSK[sent] + noise) / h, QPSK)
        return float(np.mean(detected != sent))

    def test_alamouti_doubles_the_slope_of_a_single_antenna(self):
        alamouti = [self.alamouti_ser(value) for value in self.SNR_DB]
        siso = [self.siso_ser(value) for value in self.SNR_DB]
        self.assertAlmostEqual(self.slope(siso), -1.0, delta=0.15)
        self.assertAlmostEqual(self.slope(alamouti), -2.0, delta=0.25)
        # and it is not only the slope: at the top of the range the two
        # are orders of magnitude apart
        self.assertLess(alamouti[-1], siso[-1] / 20)


class TestRegistry(unittest.TestCase):
    """The same shape as ``get_alphabet``: a name in, an object out."""

    def test_an_unknown_name_lists_the_known_ones(self):
        with self.assertRaises(KeyError) as ctx:
            get_code("turbo-space-time")
        message = str(ctx.exception)
        self.assertIn("alamouti", message)
        self.assertIn("register_code", message)

    def test_a_user_can_add_a_code(self):
        @register_code("delay-diversity-2")
        def _delay_diversity():
            direct = np.zeros((2, 2, 3), dtype=complex)
            direct[0, 0, 0] = direct[1, 0, 1] = 1.0      # antenna 0: s1, s2
            direct[0, 1, 1] = direct[1, 1, 2] = 1.0      # antenna 1: delayed
            return SpaceTimeCode("delay-diversity-2", direct,
                                 np.zeros_like(direct))
        self.assertIn("delay-diversity-2", available_codes())
        code = get_code("delay-diversity-2")
        self.assertEqual(code.minimum_rank(QPSK), 2)     # full diversity...
        self.assertFalse(code.is_orthogonal)             # ...but not linear

    def test_the_registry_entries_all_build(self):
        for name in available_codes():
            with self.subTest(code=name):
                self.assertIsInstance(get_code(name), SpaceTimeCode)


if __name__ == "__main__":
    unittest.main()
