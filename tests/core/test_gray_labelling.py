"""The Gray promise, locked end to end.

`compute_ber` expands symbol indices with `sym_2_bin` (natural binary),
and its docstring admits the count is only meaningful if the
constellation is labelled the same way. Nothing verified that promise:
the tutorials' BER curves matched theory in the tail, which is evidence
but not a lock. These tests are the lock.

Two deterministic properties close the question without Monte-Carlo:
geometric neighbours must differ by exactly one bit of their *index*
(so index == label), and a nearest-neighbour symbol error must count
exactly one bit error through `compute_ber`.
"""
import unittest

import numpy as np

from comnumpy.core.metrics import compute_ber
from comnumpy.core.utils import Constellation, get_alphabet


def hamming(a, b):
    return bin(int(a) ^ int(b)).count("1")


class TestGrayAdjacency(unittest.TestCase):
    """Nearest neighbours of a Gray alphabet differ by one index bit."""

    def nearest_pairs(self, alphabet):
        """Every (i, j) pair at the minimum geometric distance."""
        points = np.asarray(alphabet)
        distance = np.abs(points[:, None] - points[None, :])
        np.fill_diagonal(distance, np.inf)
        minimum = distance.min()
        pairs = np.argwhere(np.isclose(distance, minimum))
        return [(int(i), int(j)) for i, j in pairs]

    def test_square_qam_neighbours_differ_by_one_bit(self):
        for order in (4, 16, 64, 256):
            with self.subTest(order=order):
                alphabet = get_alphabet("QAM", order)
                for i, j in self.nearest_pairs(alphabet):
                    self.assertEqual(
                        hamming(i, j), 1,
                        f"QAM{order}: indices {i} and {j} are geometric "
                        f"neighbours but differ by {hamming(i, j)} bits -- "
                        f"compute_ber would overcount every symbol error")

    def test_psk_ring_neighbours_differ_by_one_bit(self):
        for order in (4, 8, 16):
            with self.subTest(order=order):
                alphabet = get_alphabet("PSK", order)
                for i, j in self.nearest_pairs(alphabet):
                    self.assertEqual(hamming(i, j), 1)

    def test_natural_binary_mapping_is_not_gray(self):
        """The property is the labelling, not an accident of geometry."""
        alphabet = get_alphabet("QAM", 16, type="bin")
        distances = []
        for i, j in self.nearest_pairs(alphabet):
            distances.append(hamming(i, j))
        self.assertGreater(max(distances), 1)


class TestComputeBerCountsGrayBits(unittest.TestCase):

    def test_a_nearest_neighbour_error_costs_exactly_one_bit(self):
        """The bridge between the labelling and the metric."""
        constellation = Constellation("QAM", 16)
        alphabet = np.asarray(constellation.alphabet)
        distance = np.abs(alphabet[:, None] - alphabet[None, :])
        np.fill_diagonal(distance, np.inf)
        reference = np.arange(16)
        detected = distance.argmin(axis=1)      # each symbol -> a neighbour
        ber = compute_ber(reference, detected, width=4)
        self.assertAlmostEqual(float(ber), 1.0 / 4.0)

    def test_tail_ber_matches_the_gray_closed_form(self):
        """One seeded run: measured BER within 20 % of Ps/k in the tail.

        At Eb/N0 = 11 dB a 16-QAM symbol error is almost surely a
        nearest-neighbour event, so BER = SER/4 holds tightly; a
        labelling mismatch would multiply the measured BER by ~2 and
        fail this comfortably. Seeded, hence deterministic.
        """
        from comnumpy.core import Sequential
        from comnumpy.core.channels import AWGN
        from comnumpy.core.generators import SymbolGenerator
        from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
        from comnumpy.core.utils import ebn0_to_snr_dB

        constellation = Constellation("QAM", 16)
        chain = Sequential([
            SymbolGenerator(16, name="tx"),
            SymbolMapper(constellation),
            AWGN(snr_dB=float(ebn0_to_snr_dB(11.0, bits_per_symbol=4))),
            SymbolDemapper(constellation),
        ], observations=["tx"])
        chain.seed(3)
        detected = chain(400000)
        measured = float(compute_ber(chain.observation("tx"), detected, width=4))
        theory = float(constellation.metrics(11.0, per="bit")["ber"])
        self.assertGreater(measured, 0.0, "no errors: raise N or the SNR")
        self.assertLess(abs(measured - theory) / theory, 0.20,
                        f"measured {measured:.3e} vs Gray closed form "
                        f"{theory:.3e} -- a labelling mismatch doubles this")


if __name__ == "__main__":
    unittest.main()
