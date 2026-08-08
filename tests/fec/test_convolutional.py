"""Golden tests for the convolutional encoder / Viterbi decoder (D4)."""
import unittest

import numpy as np

from comnumpy.exceptions import ShapeError
from comnumpy.fec import ConvolutionalEncoder, ViterbiDecoder
from comnumpy.fec.convolutional import _output_table, _parse_generators


def free_distance(g):
    """Minimum output weight of a diverging/remerging trellis path."""
    g, K = _parse_generators(g)
    S = 2 ** (K - 1)
    out = _output_table(g, K)
    INF = 10**9
    dist = np.full(S, INF)
    first_reg = 1 << (K - 1)
    dist[first_reg >> 1] = int(out[first_reg].sum())
    for _ in range(4 * S):
        new = dist.copy()
        for s in range(S):
            if dist[s] >= INF:
                continue
            for b in (0, 1):
                reg = (b << (K - 1)) | s
                s2 = reg >> 1
                if s2 == 0:
                    continue  # remerge handled below
                w = dist[s] + int(out[reg].sum())
                if w < new[s2]:
                    new[s2] = w
        if np.array_equal(new, dist):
            break
        dist = new
    best = INF
    for s in range(1, S):
        if s >> 1 == 0 and dist[s] < INF:  # b=0 transition back to state 0
            best = min(best, dist[s] + int(out[s].sum()))
    return best


class TestTrellisInvariants(unittest.TestCase):

    def test_free_distance_of_standard_code(self):
        """(133, 171) K=7 has free distance 10 (Proakis, Table 8.2-1)."""
        self.assertEqual(free_distance((0o133, 0o171)), 10)

    def test_free_distance_of_k3_code(self):
        """(5, 7) K=3 has free distance 5."""
        self.assertEqual(free_distance((0o5, 0o7)), 5)


class TestRoundTrip(unittest.TestCase):

    def test_noiseless_roundtrip_batched(self):
        rng = np.random.default_rng(0)
        encoder, decoder = ConvolutionalEncoder(), ViterbiDecoder()
        for shape in [(200,), (3, 120), (2, 2, 60)]:
            bits = rng.integers(0, 2, shape)
            np.testing.assert_array_equal(decoder(encoder(bits)), bits)

    def test_corrects_separated_errors(self):
        """dfree=10 corrects floor((10-1)/2)=4 well-separated hard errors."""
        rng = np.random.default_rng(1)
        bits = rng.integers(0, 2, 300)
        coded = ConvolutionalEncoder()(bits)
        for pos in (10, 150, 300, 450):
            coded[pos] ^= 1
        np.testing.assert_array_equal(ViterbiDecoder()(coded), bits)

    def test_rejects_wrong_length(self):
        with self.assertRaises(ShapeError):
            ViterbiDecoder()(np.zeros(11))


class TestCodingGain(unittest.TestCase):

    def test_soft_beats_hard_beats_uncoded(self):
        """BPSK over AWGN at Eb/N0 = 4 dB: soft < hard < uncoded BER."""
        rng = np.random.default_rng(42)
        N = 40_000
        ebn0 = 10 ** (4 / 10)
        bits = rng.integers(0, 2, N)

        encoder = ConvolutionalEncoder()
        coded = encoder(bits)

        # BPSK, unit symbol energy; rate 1/2 -> Es = Eb * R
        sigma2_coded = 1 / (2 * encoder.rate * ebn0)
        sigma2_uncoded = 1 / (2 * ebn0)

        tx = 1.0 - 2.0 * coded
        rx = tx + rng.normal(scale=np.sqrt(sigma2_coded), size=tx.shape)

        hard_in = (rx < 0).astype(int)
        ber_hard = np.mean(ViterbiDecoder(soft=False)(hard_in) != bits)

        llr = 2 * rx / sigma2_coded  # log P(0)/P(1) for BPSK 0 -> +1
        ber_soft = np.mean(ViterbiDecoder(soft=True)(llr) != bits)

        tx_u = 1.0 - 2.0 * bits
        rx_u = tx_u + rng.normal(scale=np.sqrt(sigma2_uncoded), size=tx_u.shape)
        ber_uncoded = np.mean((rx_u < 0) != bits)

        # measured with this seed: uncoded 1.31e-2, hard 4.7e-3, soft 5e-5
        # (hard-decision gain is modest at 4 dB; soft adds the usual ~2 dB)
        self.assertGreater(ber_uncoded, 5e-3)
        self.assertLess(ber_hard, ber_uncoded / 2)
        self.assertLess(ber_soft, ber_hard / 10)


class TestCodedQamChain(unittest.TestCase):
    """End-to-end D4+D12: coded 16-QAM with soft demapping."""

    def test_soft_demapper_feeds_viterbi(self):
        from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
        from comnumpy.core.utils import get_alphabet

        rng = np.random.default_rng(5)
        M, k = 16, 4
        alphabet = get_alphabet("QAM", M)
        bits = rng.integers(0, 2, 10_000)

        encoder = ConvolutionalEncoder()
        coded = encoder(bits)
        # pad the coded stream to a multiple of k, group bits MSB-first
        pad = (-len(coded)) % k
        coded_padded = np.concatenate([coded, np.zeros(pad, dtype=int)])
        weights = 1 << np.arange(k - 1, -1, -1)
        symbols = coded_padded.reshape(-1, k) @ weights

        sigma2 = 0.02
        y = alphabet[symbols] + rng.normal(scale=np.sqrt(sigma2 / 2), size=symbols.shape) \
            + 1j * rng.normal(scale=np.sqrt(sigma2 / 2), size=symbols.shape)

        llr = SymbolDemapper(alphabet, soft=True, sigma2=sigma2)(y)
        llr = llr[:len(coded)]  # drop the padding LLRs
        decoded = ViterbiDecoder(soft=True)(llr)
        ber = np.mean(decoded != bits)
        self.assertLess(ber, 1e-4)

        # SymbolMapper/Demapper consistency in the same chain, noiseless
        z = SymbolMapper(alphabet)(symbols)
        llr0 = SymbolDemapper(alphabet, soft=True)(z)[:len(coded)]
        np.testing.assert_array_equal(ViterbiDecoder(soft=True)(llr0), bits)


if __name__ == "__main__":
    unittest.main()
