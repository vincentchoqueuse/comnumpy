"""Tests for the LDPC encoder and min-sum decoder (decision D5)."""
import unittest

import numpy as np

from comnumpy.exceptions import ShapeError
from comnumpy.fec import LDPCDecoder, LDPCEncoder, make_gallager_parity_check

HAMMING_H = np.array([[1, 1, 0, 1, 1, 0, 0],
                      [1, 0, 1, 1, 0, 1, 0],
                      [0, 1, 1, 1, 0, 0, 1]])


class TestGallagerConstruction(unittest.TestCase):

    def test_regularity(self):
        H = make_gallager_parity_check(120, d_v=3, d_c=6, seed=0)
        self.assertEqual(H.shape, (60, 120))
        np.testing.assert_array_equal(H.sum(axis=0), 3)
        np.testing.assert_array_equal(H.sum(axis=1), 6)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            make_gallager_parity_check(121, d_v=3, d_c=6)
        with self.assertRaises(ValueError):
            make_gallager_parity_check(120, d_v=6, d_c=6)


class TestLDPCEncoder(unittest.TestCase):

    def test_every_codeword_satisfies_H(self):
        H = make_gallager_parity_check(60, seed=1)
        encoder = LDPCEncoder(H)
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, (5, 4, encoder.k))
        c = encoder(bits)
        self.assertEqual(c.shape, (5, 4, 60))
        np.testing.assert_array_equal((c @ H.T) % 2, 0)

    def test_systematic_positions_carry_the_information(self):
        encoder = LDPCEncoder(HAMMING_H)
        bits = np.array([1, 0, 1, 1])
        c = encoder(bits)
        np.testing.assert_array_equal(c[encoder.free_cols], bits)

    def test_rank_deficient_H_increases_k(self):
        H = np.vstack([HAMMING_H, HAMMING_H[0] ^ HAMMING_H[1]])
        encoder = LDPCEncoder(H)
        self.assertEqual(encoder.k, 4)  # dependent row ignored

    def test_wrong_length_raises(self):
        with self.assertRaises(ShapeError):
            LDPCEncoder(HAMMING_H)(np.zeros(7))


class TestLDPCDecoder(unittest.TestCase):

    def _llr(self, c, sigma2, rng):
        y = (1.0 - 2.0 * c) + rng.normal(scale=np.sqrt(sigma2), size=c.shape)
        return 2.0 * y / sigma2

    def test_noiseless_roundtrip_batch(self):
        H = make_gallager_parity_check(60, seed=1)
        encoder, decoder = LDPCEncoder(H), LDPCDecoder(H)
        rng = np.random.default_rng(2)
        bits = rng.integers(0, 2, (3, 2, encoder.k))
        llr = 8.0 * (1.0 - 2.0 * encoder(bits))
        np.testing.assert_array_equal(decoder(llr), bits)

    def test_output_codeword(self):
        encoder = LDPCEncoder(HAMMING_H)
        decoder = LDPCDecoder(HAMMING_H, output="codeword")
        c = encoder(np.array([1, 0, 1, 1]))
        np.testing.assert_array_equal(decoder(8.0 * (1.0 - 2.0 * c)), c)

    def test_empty_row_raises(self):
        H = np.vstack([HAMMING_H, np.zeros(7, dtype=int)])
        with self.assertRaises(ValueError):
            LDPCDecoder(H)

    def test_wrong_length_raises(self):
        with self.assertRaises(ShapeError):
            LDPCDecoder(HAMMING_H)(np.zeros(6))

    def test_coding_gain_golden(self):
        """(3,6) Gallager n=240, Eb/N0 = 3 dB: >10x below uncoded BPSK."""
        H = make_gallager_parity_check(240, 3, 6, seed=7)
        encoder = LDPCEncoder(H)
        decoder = LDPCDecoder(H, n_iter=50, alpha=0.75)
        rng = np.random.default_rng(123)
        ebn0 = 10 ** (3.0 / 10)
        sigma2 = 1 / (2 * encoder.rate * ebn0)

        bits = rng.integers(0, 2, (200, encoder.k))
        ber = np.mean(decoder(self._llr(encoder(bits), sigma2, rng)) != bits)

        sigma2_u = 1 / (2 * ebn0)  # uncoded BPSK at the same Eb/N0
        yu = (1.0 - 2.0 * bits) + rng.normal(scale=np.sqrt(sigma2_u),
                                             size=bits.shape)
        ber_uncoded = np.mean((yu < 0) != bits)
        self.assertLess(ber, ber_uncoded / 10,
                        f"coded {ber} vs uncoded {ber_uncoded}")

    def test_normalized_beats_plain_min_sum(self):
        """alpha=0.75 must not be worse than plain min-sum (sanity)."""
        H = make_gallager_parity_check(120, 3, 6, seed=3)
        encoder = LDPCEncoder(H)
        rng = np.random.default_rng(9)
        sigma2 = 1 / (2 * encoder.rate * 10 ** (2.5 / 10))
        bits = rng.integers(0, 2, (300, encoder.k))
        llr = self._llr(encoder(bits), sigma2, rng)
        ber_plain = np.mean(LDPCDecoder(H, n_iter=50)(llr) != bits)
        ber_norm = np.mean(
            LDPCDecoder(H, n_iter=50, alpha=0.75)(llr) != bits)
        self.assertLessEqual(ber_norm, ber_plain)


if __name__ == "__main__":
    unittest.main()
