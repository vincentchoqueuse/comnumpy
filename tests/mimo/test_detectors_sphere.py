r"""The sphere decoder is maximum likelihood, or it is nothing.

A tree search that prunes is only worth having if what survives the
pruning is *exactly* what the exhaustive search would have returned.
So the first family of tests confronts the two detectors symbol by
symbol, over channels and constellations chosen to make them disagree
if anything is wrong -- ill-conditioned matrices, low SNR, non-square
alphabets -- and the second checks the property the whole construction
exists for: the number of visited nodes collapses when the SNR grows,
far below the exhaustive count.
"""
import unittest

import numpy as np

from comnumpy.core.utils import get_alphabet
from comnumpy.mimo.detectors import MaximumLikelihoodDetector, SphereDecoder
from comnumpy.mimo.utils import rayleigh_channel


def received(alphabet, H, sent, sigma, seed=0):
    rng = np.random.default_rng(seed)
    n_rx, _ = H.shape
    noise = (rng.standard_normal((n_rx, sent.shape[-1]))
             + 1j * rng.standard_normal((n_rx, sent.shape[-1])))
    return H @ alphabet[sent] + sigma / np.sqrt(2) * noise


class TestItIsExactlyMaximumLikelihood(unittest.TestCase):

    def confront(self, family, order, n_tx, n_rx, sigma, seed=0, n=200):
        alphabet = get_alphabet(family, order)
        H = rayleigh_channel(n_rx, n_tx, seed=seed)
        rng = np.random.default_rng(seed + 100)
        sent = rng.integers(0, order, size=(n_tx, n))
        Y = received(alphabet, H, sent, sigma, seed=seed + 200)
        sphere = SphereDecoder(alphabet, H=H)
        exhaustive = MaximumLikelihoodDetector(alphabet, H=H)
        np.testing.assert_array_equal(sphere(Y), exhaustive(Y))
        return sphere

    def test_it_agrees_on_every_shape_and_alphabet(self):
        for family, order, n_tx, n_rx in (("PSK", 4, 2, 2), ("PSK", 4, 2, 4),
                                          ("PSK", 8, 3, 3), ("QAM", 16, 2, 2),
                                          ("QAM", 16, 3, 4), ("PAM", 4, 3, 3)):
            with self.subTest(alphabet=f"{family}-{order}",
                              channel=f"{n_rx}x{n_tx}"):
                self.confront(family, order, n_tx, n_rx, sigma=0.5)

    def test_it_agrees_where_pruning_is_hardest(self):
        """Low SNR: the sphere stays wide and almost nothing is cut."""
        for sigma in (1.0, 2.0, 4.0):
            with self.subTest(sigma=sigma):
                self.confront("QAM", 16, 3, 3, sigma=sigma, seed=7)

    def test_it_agrees_on_an_ill_conditioned_channel(self):
        """Two nearly parallel columns: the tree is deep and narrow."""
        alphabet = get_alphabet("QAM", 16)
        H = np.array([[1.0, 1.0 - 1e-3], [1e-3, 1e-3 + 1e-6]], dtype=complex)
        rng = np.random.default_rng(3)
        sent = rng.integers(0, 16, size=(2, 100))
        Y = received(alphabet, H, sent, sigma=0.05, seed=4)
        np.testing.assert_array_equal(
            SphereDecoder(alphabet, H=H)(Y),
            MaximumLikelihoodDetector(alphabet, H=H)(Y))

    def test_a_noiseless_channel_is_decoded_exactly(self):
        alphabet = get_alphabet("QAM", 16)
        H = rayleigh_channel(4, 4, seed=5)
        rng = np.random.default_rng(6)
        sent = rng.integers(0, 16, size=(4, 100))
        np.testing.assert_array_equal(
            SphereDecoder(alphabet, H=H)(H @ alphabet[sent]), sent)


class TestWhatThePruningBuys(unittest.TestCase):

    def nodes(self, snr_dB, order=16, n=4, count=200):
        alphabet = get_alphabet("QAM", order)
        H = rayleigh_channel(n, n, seed=2)
        rng = np.random.default_rng(11)
        sent = rng.integers(0, order, size=(n, count))
        sigma = np.sqrt(n * 10 ** (-snr_dB / 10))
        decoder = SphereDecoder(alphabet, H=H)
        decoder(received(alphabet, H, sent, sigma, seed=12))
        return decoder.nodes_

    def test_the_search_collapses_as_the_snr_grows(self):
        counts = [self.nodes(snr_dB) for snr_dB in (5, 15, 25)]
        self.assertTrue(counts[0] > counts[1] > counts[2], counts)
        # at high SNR the successive-cancellation path is already ML:
        # one node per layer, and nothing else survives the first bound
        self.assertLess(counts[-1], 8.0, counts)

    def test_it_visits_a_vanishing_fraction_of_the_candidates(self):
        self.assertLess(self.nodes(15) / 16 ** 4, 1e-3)

    def test_the_node_count_is_reported_per_vector(self):
        alphabet = get_alphabet("PSK", 4)
        H = rayleigh_channel(2, 2, seed=0)
        decoder = SphereDecoder(alphabet, H=H)
        decoder(H @ alphabet[np.zeros((2, 50), dtype=int)])
        self.assertGreaterEqual(decoder.nodes_, 2.0)      # one per layer
        self.assertLessEqual(decoder.nodes_, 4 ** 2)


class TestGuards(unittest.TestCase):

    def test_more_streams_than_antennas_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            SphereDecoder(get_alphabet("PSK", 4),
                          H=rayleigh_channel(2, 3, seed=0))(np.zeros((2, 5),
                                                                     dtype=complex))
        self.assertIn("underdetermined", str(ctx.exception))

    def test_a_rank_deficient_channel_is_refused(self):
        H = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)
        with self.assertRaises(ValueError) as ctx:
            SphereDecoder(get_alphabet("PSK", 4), H=H)(np.zeros((2, 5),
                                                                dtype=complex))
        self.assertIn("rank-deficient", str(ctx.exception))

    def test_a_missing_channel_names_the_parameter(self):
        with self.assertRaises(ValueError) as ctx:
            SphereDecoder(get_alphabet("PSK", 4))(np.zeros((2, 5),
                                                           dtype=complex))
        self.assertIn("H=None", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
